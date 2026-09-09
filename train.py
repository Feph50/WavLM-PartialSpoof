import argparse
import os
import re
import sys
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any

# Suppress noisy library deprecation warnings (weight_norm, torchaudio, etc.)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from src.dataset import PartialSpoofDataModule
from src.model import WavLMConformer, WavLMConformerDiarization
from src.criterion import TotalLoss, TotalDiarizationLoss, JERMetric, JIBonaMetric
from src.pipeline import WavLMConformerPipeline


class TeeLogger:
    """
    Duplicates stdout/stderr to both console and a log file.
    """

    def __init__(self, log_filepath: str) -> None:
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


class ExperimentRunner:
    """
    Encapsulates Experiment Orchestration, Run Folder Management (run1, run2,...),
    TQDM Progress Bar Logging, and Execution.
    """

    def __init__(self, config_path: str, cli_args: argparse.Namespace) -> None:
        self.config_path = config_path
        self.args = cli_args
        self.cfg = self._load_and_merge_config()

    def _load_and_merge_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if self.args.debug:
            cfg["data"]["debug_mode"] = True
            cfg["data"]["max_samples"] = self.args.max_samples or 50
        if self.args.batch_size is not None:
            cfg["data"]["batch_size"] = self.args.batch_size
        if self.args.lr is not None:
            cfg["training"]["learning_rate"] = self.args.lr
        if self.args.max_epochs is not None:
            cfg["training"]["max_epochs"] = self.args.max_epochs
        if self.args.devices is not None:
            cfg["training"]["devices"] = self.args.devices
        if self.args.seed is not None:
            cfg["training"]["seed"] = self.args.seed

        return cfg

    @staticmethod
    def get_next_run_dir(base_exp_dir: str) -> str:
        """Finds next available run directory: run1, run2, run3, ..."""
        os.makedirs(base_exp_dir, exist_ok=True)
        existing_runs = []
        for name in os.listdir(base_exp_dir):
            match = re.match(r"^run(\d+)$", name)
            if match and os.path.isdir(os.path.join(base_exp_dir, name)):
                existing_runs.append(int(match.group(1)))

        next_idx = max(existing_runs) + 1 if existing_runs else 1
        run_dir = os.path.join(base_exp_dir, f"run{next_idx}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    @staticmethod
    def run_sanity_check() -> None:
        """Sanity check for TotalLoss, TotalDiarizationLoss, Gradient Flow, and JER Metric."""
        print("\n=== [Sanity Check 1] Testing Localization Contrastive Segment Loss & Gradient Flow ===")
        loss_fn = TotalLoss(lambda_contrastive=0.5, margin=1.0, alpha_intra=1.0, beta_inter=1.0)

        b, s, d = 4, 10, 1024
        logits = torch.randn(b, s, 2, requires_grad=True)
        embeddings = torch.randn(b, s, d, requires_grad=True)

        targets = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ])

        mask = torch.tensor([
            [True] * 10,
            [True] * 7 + [False] * 3,
            [True] * 5 + [False] * 5,
            [True] * 10,
        ])

        total_loss, loss_dict = loss_fn(logits, embeddings, targets, mask=mask)

        print(f"  Total Loss       : {total_loss.item():.4f}")
        print(f"  BCE Loss         : {loss_dict['loss_bce'].item():.4f}")
        print(f"  Contrastive Loss : {loss_dict['loss_contrastive'].item():.4f}")
        print(f"  Intra Loss       : {loss_dict['loss_intra'].item():.4f}")
        print(f"  Inter Loss       : {loss_dict['loss_inter'].item():.4f}")

        total_loss.backward()
        assert logits.grad is not None and embeddings.grad is not None, "Gradients not computed!"
        print("✓ Localization loss sanity check passed successfully!")

        print("\n=== [Sanity Check 2] Testing Diarization Loss & JER Metric ===")
        dia_loss_fn = TotalDiarizationLoss(loc_loss=TotalLoss(), lambda_dia=1.0, spoof_only=True)
        loc_logits = torch.randn(b, s, 2, requires_grad=True)
        loc_embs = torch.randn(b, s, d, requires_grad=True)
        dia_logits = torch.randn(b, s, 10, requires_grad=True)

        joint_loss, joint_dict = dia_loss_fn(
            loc_logits=loc_logits,
            loc_embeddings=loc_embs,
            loc_targets=targets,
            dia_logits=dia_logits,
            dia_targets=targets,
            mask=mask,
        )
        print(f"  Joint Total Loss : {joint_loss.item():.4f}")
        print(f"  Dia CE Loss      : {joint_dict['loss_dia'].item():.4f}")
        joint_loss.backward()
        assert loc_logits.grad is not None and dia_logits.grad is not None, "Diarization gradients not computed!"
        print("✓ Diarization loss backward passed successfully!")

        print("\n=== [Sanity Check 3] Testing Diarization Metrics (Hungarian Matching & Oracle VAD) ===")
        # Ground truth: 1 = bonafide, 2 = attack A01, 3 = attack A02, 0 = silence
        # Hyp: -1 = bonafide prediction, 5 = predicted cluster for A01, 6 = predicted cluster for A02
        jer_metric = JERMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)
        ji_bona_metric = JIBonaMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)

        hyp_perfect = torch.tensor([
            [-1, -1, 5, 5, -1],   # predicted cluster 5 matches gt attack 2; frame 4 is silence
            [-1, -1, -1, -1, -1],  # all bonafide predicted
        ])
        gt_perfect = torch.tensor([
            [1, 1, 2, 2, 0],      # 1=bona, 2=attack A01, 0=silence
            [1, 1, 1, 1, 0],      # 1=bona, 0=silence
        ])
        jer_metric.update(hyp_perfect, gt_perfect)
        ji_bona_metric.update(hyp_perfect, gt_perfect)
        jer_score = jer_metric.compute()
        ji_score = ji_bona_metric.compute()
        print(f"  JER_spoof on perfect match (with permutation): {jer_score:.2f} % (Expected: 0.00 %)")
        print(f"  JI_bona on perfect match: {ji_score:.2f} % (Expected: 0.00 %)")
        assert jer_score == 0.0, f"JER should be 0.0, got {jer_score}"
        assert ji_score == 0.0, f"JI_bona should be 0.0, got {ji_score}"
        print("✓ Diarization metrics passed successfully!\n")

    def run(self) -> None:
        if self.args.test_loss:
            self.run_sanity_check()
            return

        # 1. Setup Run Directory (run1, run2, ...)
        log_dir = self.cfg["logging"]["log_dir"]
        exp_name = self.cfg["logging"]["experiment_name"]
        base_exp_dir = os.path.join(log_dir, exp_name)

        if self.args.test_only and self.args.ckpt_path:
            ckpt_dir = os.path.dirname(os.path.abspath(self.args.ckpt_path))
            run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir
        else:
            run_dir = self.get_next_run_dir(base_exp_dir)

        # 2. Redirect stdout/stderr to train.log in this run_dir
        log_file_path = os.path.join(run_dir, "train.log")
        tee_logger = TeeLogger(log_file_path)
        sys.stdout = tee_logger
        sys.stderr = tee_logger

        # 3. Save clean hparams.yaml in run_dir
        hparams_path = os.path.join(run_dir, "hparams.yaml")
        with open(hparams_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

        print("=" * 60)
        print(f"[*] Experiment Run Directory : {run_dir}")
        print(f"[*] Hyperparameters Saved    : {hparams_path}")
        print(f"[*] Terminal Log File        : {log_file_path}")
        print("=" * 60 + "\n")

        # 4. Seed
        seed = self.cfg["training"].get("seed", 42)
        L.seed_everything(seed, workers=True)

        # 5. DataModule
        print("[*] Initializing DataModule...")
        data_module = PartialSpoofDataModule(**self.cfg["data"])

        # 6. Model & Loss
        print("[*] Initializing Localization Backbone Model...")
        loc_model = WavLMConformer(**self.cfg["model"])

        print("[*] Initializing Loss...")
        loc_loss = TotalLoss(**self.cfg["loss"])

        if self.cfg.get("diarization", {}).get("enabled", False):
            print("[*] Initializing Spoof Diarization Baseline (Two-Branch Architecture)...")
            dia_cfg = self.cfg["diarization"]
            model = WavLMConformerDiarization(
                loc_model=loc_model,
                num_spoof_classes=dia_cfg.get("num_spoof_classes", 10),
                hidden_dim=dia_cfg.get("hidden_dim", 256),
                dia_emb_dim=dia_cfg.get("dia_emb_dim", 128),
                use_loc_guidance=dia_cfg.get("use_loc_guidance", True),
                dropout=dia_cfg.get("dropout", 0.1),
            )
            loss_fn = TotalDiarizationLoss(
                loc_loss=loc_loss,
                lambda_dia=dia_cfg.get("lambda_dia", 1.0),
                spoof_only=dia_cfg.get("spoof_only", True),
            )
        else:
            model = loc_model
            loss_fn = loc_loss

        # 7. Lightning Pipeline Module
        pipeline = WavLMConformerPipeline(
            model=model,
            loss_fn=loss_fn,
            learning_rate=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"]["weight_decay"],
            scheduler_type=self.cfg["training"]["scheduler"],
            warmup_epochs=self.cfg["training"]["warmup_epochs"],
            max_epochs=self.cfg["training"]["max_epochs"],
            result_save_dir=run_dir,
        )

        # 8. Checkpoint callback & TQDM Progress Bar
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_eer_{epoch:02d}_{val_eer:.2f}",
            monitor="val_eer",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        progress_bar = TQDMProgressBar(refresh_rate=1)

        # 9. Lightning Trainer
        accelerator = self.cfg["training"].get("accelerator", "auto")
        if not torch.cuda.is_available():
            accelerator = "cpu"

        trainer = L.Trainer(
            max_epochs=self.cfg["training"]["max_epochs"],
            accelerator=accelerator,
            devices=self.cfg["training"].get("devices", 1) if accelerator != "cpu" else "auto",
            accumulate_grad_batches=self.cfg["training"].get("accumulate_grad_batches", 1),
            precision=self.cfg["training"].get("precision", "32-true"),
            gradient_clip_val=self.cfg["training"].get("gradient_clip_val", 1.0),
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=[checkpoint_callback, progress_bar],
        )

        # 10. Fit or Test
        if not self.args.test_only:
            print("\n=== Starting Training ===")
            trainer.fit(
                model=pipeline,
                datamodule=data_module,
                ckpt_path=self.args.ckpt_path,
            )

            print("\n=== Starting Post-Training Evaluation on Test Split ===")
            best_ckpt = checkpoint_callback.best_model_path or "best"
            trainer.test(model=pipeline, datamodule=data_module, ckpt_path=best_ckpt)
        else:
            print("\n=== Running Test Evaluation Only ===")
            assert self.args.ckpt_path is not None, "Must provide --ckpt_path for test_only mode"
            trainer.test(model=pipeline, datamodule=data_module, ckpt_path=self.args.ckpt_path)

        print(f"\n[✓] Run completed. All logs saved to: {log_file_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WavLM-Conformer with Contrastive Segment Loss (OOP)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config/baseline.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--test-loss", action="store_true", help="Run sanity check on loss & gradients")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with limited samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split in debug mode")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--devices", type=int, default=None, help="Number of GPU devices")
    parser.add_argument("--test_only", action="store_true", help="Run testing only")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path for resume/testing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle("python3 src/embed.py --model qwen --all")
    
    args = parse_args()
    runner = ExperimentRunner(config_path=args.config, cli_args=args)
    runner.run()