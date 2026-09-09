import os
from typing import Optional, Dict, Any, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from src.model import WavLMConformer, WavLMConformerDiarization
from src.criterion import (
    TotalLoss,
    TotalDiarizationLoss,
    EERMetric,
    F1Metric,
    JERMetric,
    JIBonaMetric,
)


class WavLMConformerPipeline(LightningModule):
    """
    LightningModule encapsulating the full training, validation, and testing pipeline
    for WavLM-Conformer with Contrastive Segment Loss and Diarization.
    """

    def __init__(
        self,
        model: WavLMConformer,
        loss_fn: TotalLoss,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 2,
        max_epochs: int = 30,
        result_save_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.result_save_dir = result_save_dir

        # Check if diarization branch is active
        self.is_diarization = hasattr(self.model, "diarization_head")

        # Trackers
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_eer = EERMetric()
        self.val_eer_best = MinMetric()
        self.val_f1_acc = F1Metric()

        self.test_eer = EERMetric()
        self.test_f1_acc = F1Metric()

        if self.is_diarization:
            self.val_ji_bona = JIBonaMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)
            self.val_jer = JERMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)
            self.test_ji_bona = JIBonaMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)
            self.test_jer = JERMetric(percent=True, bonafide_pred_label=-1, bonafide_gt_label=1, oracle_vad=True)

    @staticmethod
    def get_label_mask(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
        """Constructs boolean mask [Batch, T_segments] for valid non-padded tokens."""
        batch_size, max_len = labels.shape
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=labels.device)
        for i, length in enumerate(label_lengths):
            mask[i, : int(length.item())] = True
        return mask

    @staticmethod
    def flatten_valid_predictions(
        preds: torch.Tensor, labels: torch.Tensor, label_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flattens non-padded predictions and labels for evaluation metrics."""
        pred_list, label_list = [], []
        for i, length in enumerate(label_lengths):
            l = int(length.item())
            pred_list.append(preds[i, :l])
            label_list.append(labels[i, :l])

        preds_flat = torch.cat(pred_list, dim=0)
        labels_flat = torch.cat(label_list, dim=0).long()
        return preds_flat, labels_flat

    @staticmethod
    def _align_temporal_lengths(
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aligns temporal dimensions between predictions and labels."""
        min_len = min(logits.size(1), labels.size(1))
        return (
            logits[:, :min_len],
            embeddings[:, :min_len],
            labels[:, :min_len],
            mask[:, :min_len],
        )

    @staticmethod
    def map_raw_labels(raw_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps raw segment labels to localization targets, diarization targets, and VAD speech mask.
        Supports both legacy binary labels (0/1) and detailed v1.3 labels.

        Detailed v1.3 Encoding:
          0:         nonspeech
          1:         bonafide
          2..7:      A01..A06 (known spoof attacks in train/dev)
          8..20:     A07..A19 (unseen spoof attacks in eval)
          100:       nonmix (concatenated boundary part / ConP)
          101:       nonbona (nonspeech pause in bonafide)
          102..120:  nonA01..nonA19 (nonspeech pause in spoof)
        """
        is_detailed = torch.any(raw_labels > 1)
        if not is_detailed:
            loc_targets = raw_labels.long()
            dia_targets = torch.zeros_like(raw_labels, dtype=torch.long)
            speech_mask = torch.ones_like(raw_labels, dtype=torch.bool)
            return loc_targets, dia_targets, speech_mask

        # Bona fide is label 1
        is_bona = (raw_labels == 1)
        # Attacks are 2..20 (A01..A19) and 100 (ConP)
        is_attack = ((raw_labels >= 2) & (raw_labels <= 20)) | (raw_labels == 100)
        # Nonspeech are 0, 101, 102..120
        is_nonspeech = (raw_labels == 0) | ((raw_labels >= 101) & (raw_labels <= 120))

        # Binary localization target: 1 for bona fide, 0 for spoof
        loc_targets = torch.zeros_like(raw_labels, dtype=torch.long)
        loc_targets[is_bona | (raw_labels == 101)] = 1
        loc_targets[is_attack] = 0

        # Multi-class diarization target for training:
        # A01..A06 (2..7) -> 0..5
        # ConP (100) -> 6
        dia_targets = torch.full_like(raw_labels, fill_value=-100, dtype=torch.long)
        known_mask = (raw_labels >= 2) & (raw_labels <= 7)
        dia_targets[known_mask] = (raw_labels[known_mask] - 2).long()
        dia_targets[raw_labels == 100] = 6

        # Speech mask (Oracle VAD): True for speech and boundary (bona fide, attacks, ConP), False for nonspeech silences
        speech_mask = ~is_nonspeech

        return loc_targets, dia_targets, speech_mask

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def training_step(
        self, batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        utt_ids, inputs, labels, label_lengths = batch
        model_out = self.forward(inputs)
        pad_mask = self.get_label_mask(labels, label_lengths)

        if self.is_diarization:
            loc_logits, loc_embeddings, dia_logits, dia_embeddings = model_out
            min_len = min(loc_logits.size(1), dia_logits.size(1), labels.size(1))
            loc_logits = loc_logits[:, :min_len]
            loc_embeddings = loc_embeddings[:, :min_len]
            dia_logits = dia_logits[:, :min_len]
            labels = labels[:, :min_len]
            pad_mask = pad_mask[:, :min_len]

            loc_targets, dia_targets, speech_mask = self.map_raw_labels(labels)
            mask = pad_mask & speech_mask

            loss, loss_dict = self.loss_fn(
                loc_logits=loc_logits,
                loc_embeddings=loc_embeddings,
                loc_targets=loc_targets,
                dia_logits=dia_logits,
                dia_targets=dia_targets,
                mask=mask,
            )
            self.train_loss.update(loss)
            self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("bce", loss_dict["loss_bce"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("cont", loss_dict["loss_contrastive"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            if "loss_dia" in loss_dict:
                self.log("dia", loss_dict["loss_dia"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss
        else:
            logits, embeddings = model_out
            logits, embeddings, labels, pad_mask = self._align_temporal_lengths(logits, embeddings, labels, pad_mask)
            loss, loss_dict = self.loss_fn(logits, embeddings, labels, mask=pad_mask)

            self.train_loss.update(loss)
            self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("bce", loss_dict["loss_bce"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("cont", loss_dict["loss_contrastive"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()

    def validation_step(
        self, batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        utt_ids, inputs, labels, label_lengths = batch
        model_out = self.forward(inputs)
        pad_mask = self.get_label_mask(labels, label_lengths)

        if self.is_diarization:
            loc_logits, loc_embeddings, dia_logits, dia_embeddings = model_out
            min_len = min(loc_logits.size(1), dia_logits.size(1), labels.size(1))
            loc_logits = loc_logits[:, :min_len]
            loc_embeddings = loc_embeddings[:, :min_len]
            dia_logits = dia_logits[:, :min_len]
            labels = labels[:, :min_len]
            pad_mask = pad_mask[:, :min_len]

            loc_targets, dia_targets, speech_mask = self.map_raw_labels(labels)
            mask = pad_mask & speech_mask

            loss, _ = self.loss_fn(
                loc_logits=loc_logits,
                loc_embeddings=loc_embeddings,
                loc_targets=loc_targets,
                dia_logits=dia_logits,
                dia_targets=dia_targets,
                mask=mask,
            )
            preds_flat, labels_flat = self.flatten_valid_predictions(loc_logits, loc_targets, label_lengths)

            self.val_loss.update(loss)
            probs_fake = torch.softmax(preds_flat, dim=-1)[:, 0]
            fake_targets = (labels_flat == 0).long()

            self.val_eer.update(probs_fake, fake_targets)
            self.val_f1_acc.update(preds_flat, labels_flat)

            # Diarization evaluation using LCM fusion
            dia_preds = dia_logits.argmax(dim=-1)
            fused_preds = self.model.apply_lcm(loc_logits, dia_preds, bonafide_idx=1, bonafide_label=-1)
            self.val_ji_bona.update(fused_preds, labels, label_lengths=label_lengths)
            self.val_jer.update(fused_preds, labels, label_lengths=label_lengths)
        else:
            logits, embeddings = model_out
            logits, embeddings, labels, pad_mask = self._align_temporal_lengths(logits, embeddings, labels, pad_mask)
            loss, _ = self.loss_fn(logits, embeddings, labels, mask=pad_mask)

            preds_flat, labels_flat = self.flatten_valid_predictions(logits, labels, label_lengths)

            self.val_loss.update(loss)
            probs_fake = torch.softmax(preds_flat, dim=-1)[:, 0]
            fake_targets = (labels_flat == 0).long()

            self.val_eer.update(probs_fake, fake_targets)
            self.val_f1_acc.update(preds_flat, labels_flat)

    def on_validation_epoch_end(self) -> None:
        val_loss_val = self.val_loss.compute().item()
        eer, thresh = self.val_eer.compute()
        acc, f1 = self.val_f1_acc.compute()
        self.val_eer_best.update(eer)
        best_eer = self.val_eer_best.compute().item()

        self.log("val_loss", val_loss_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.is_diarization:
            ji_bona = self.val_ji_bona.compute()
            jer_spoof = self.val_jer.compute()
            self.log("val_ji_bona", ji_bona, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_jer", jer_spoof, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.val_ji_bona.reset()
            self.val_jer.reset()

        self.val_loss.reset()
        self.val_eer.reset()
        self.val_f1_acc.reset()

    def test_step(
        self, batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        utt_ids, inputs, labels, label_lengths = batch
        model_out = self.forward(inputs)
        pad_mask = self.get_label_mask(labels, label_lengths)

        if self.is_diarization:
            loc_logits, loc_embeddings, dia_logits, dia_embeddings = model_out
            min_len = min(loc_logits.size(1), dia_logits.size(1), labels.size(1))
            loc_logits = loc_logits[:, :min_len]
            loc_embeddings = loc_embeddings[:, :min_len]
            dia_logits = dia_logits[:, :min_len]
            labels = labels[:, :min_len]
            pad_mask = pad_mask[:, :min_len]

            loc_targets, dia_targets, speech_mask = self.map_raw_labels(labels)
            mask = pad_mask & speech_mask

            loss, _ = self.loss_fn(
                loc_logits=loc_logits,
                loc_embeddings=loc_embeddings,
                loc_targets=loc_targets,
                dia_logits=dia_logits,
                dia_targets=dia_targets,
                mask=mask,
            )
            preds_flat, labels_flat = self.flatten_valid_predictions(loc_logits, loc_targets, label_lengths)

            self.test_loss.update(loss)
            probs_fake = torch.softmax(preds_flat, dim=-1)[:, 0]
            fake_targets = (labels_flat == 0).long()

            self.test_eer.update(probs_fake, fake_targets)
            self.test_f1_acc.update(preds_flat, labels_flat)

            dia_preds = dia_logits.argmax(dim=-1)
            fused_preds = self.model.apply_lcm(loc_logits, dia_preds, bonafide_idx=1, bonafide_label=-1)
            self.test_ji_bona.update(fused_preds, labels, label_lengths=label_lengths)
            self.test_jer.update(fused_preds, labels, label_lengths=label_lengths)
        else:
            logits, embeddings = model_out
            logits, embeddings, labels, pad_mask = self._align_temporal_lengths(logits, embeddings, labels, pad_mask)
            loss, _ = self.loss_fn(logits, embeddings, labels, mask=pad_mask)

            preds_flat, labels_flat = self.flatten_valid_predictions(logits, labels, label_lengths)

            self.test_loss.update(loss)
            probs_fake = torch.softmax(preds_flat, dim=-1)[:, 0]
            fake_targets = (labels_flat == 0).long()

            self.test_eer.update(probs_fake, fake_targets)
            self.test_f1_acc.update(preds_flat, labels_flat)

    def on_test_epoch_end(self) -> None:
        test_loss_val = self.test_loss.compute().item()
        eer, thresh = self.test_eer.compute()
        acc, f1 = self.test_f1_acc.compute()
        ji_val = self.test_ji_bona.compute() if self.is_diarization else None
        jer_val = self.test_jer.compute() if self.is_diarization else None

        print("\n" + "=" * 60)
        print("=== TEST EVALUATION RESULTS ===")
        print(f"  Test Loss              : {test_loss_val:.4f}")
        print(f"  Equal Error Rate (EER) : {eer:.2f} % (Threshold: {thresh:.4f})")
        print(f"  Segment Accuracy       : {acc:.2f} %")
        print(f"  Segment F1-Score       : {f1:.2f} %")
        if ji_val is not None:
            print(f"  JI_bona                : {ji_val:.2f} %")
        if jer_val is not None:
            print(f"  JER_spoof              : {jer_val:.2f} %")
        print("=" * 60 + "\n", flush=True)

        if self.result_save_dir is not None:
            os.makedirs(self.result_save_dir, exist_ok=True)
            result_file = os.path.join(self.result_save_dir, "test_results.txt")
            with open(result_file, "w") as f:
                f.write(f"Test_Loss: {test_loss_val:.4f}\n")
                f.write(f"EER: {eer:.4f}\n")
                f.write(f"Threshold: {thresh:.4f}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"F1: {f1:.4f}\n")
                if ji_val is not None:
                    f.write(f"JI_bona: {ji_val:.4f}\n")
                if jer_val is not None:
                    f.write(f"JER_spoof: {jer_val:.4f}\n")
            print(f"Results saved to: {result_file}")

        self.test_loss.reset()
        self.test_eer.reset()
        self.test_f1_acc.reset()
        if self.is_diarization:
            self.test_ji_bona.reset()
            self.test_jer.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        # Filter trainable parameters (SSL encoder might be frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        if self.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}