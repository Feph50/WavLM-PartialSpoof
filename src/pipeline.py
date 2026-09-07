import os
from typing import Optional, Dict, Any, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from src.model import WavLMConformer
from src.criterion import TotalLoss, EERMetric, F1Metric


class WavLMConformerPipeline(LightningModule):
    """
    LightningModule encapsulating the full training, validation, and testing pipeline
    for WavLM-Conformer with Contrastive Segment Loss.
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

        # Trackers
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_eer = EERMetric()
        self.val_eer_best = MinMetric()
        self.val_f1_acc = F1Metric()

        self.test_eer = EERMetric()
        self.test_f1_acc = F1Metric()

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(
        self, batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        utt_ids, inputs, labels, label_lengths = batch
        logits, embeddings = self.forward(inputs)

        mask = self.get_label_mask(labels, label_lengths)
        logits, embeddings, labels, mask = self._align_temporal_lengths(logits, embeddings, labels, mask)
        loss, loss_dict = self.loss_fn(logits, embeddings, labels, mask=mask)

        # Track metric & progress bar
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
        logits, embeddings = self.forward(inputs)

        mask = self.get_label_mask(labels, label_lengths)
        logits, embeddings, labels, mask = self._align_temporal_lengths(logits, embeddings, labels, mask)
        loss, _ = self.loss_fn(logits, embeddings, labels, mask=mask)

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

        self.val_loss.reset()
        self.val_eer.reset()
        self.val_f1_acc.reset()

    def test_step(
        self, batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        utt_ids, inputs, labels, label_lengths = batch
        logits, embeddings = self.forward(inputs)

        mask = self.get_label_mask(labels, label_lengths)
        logits, embeddings, labels, mask = self._align_temporal_lengths(logits, embeddings, labels, mask)
        loss, _ = self.loss_fn(logits, embeddings, labels, mask=mask)

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

        print("\n" + "=" * 60)
        print("=== TEST EVALUATION RESULTS ===")
        print(f"  Test Loss              : {test_loss_val:.4f}")
        print(f"  Equal Error Rate (EER) : {eer:.2f} % (Threshold: {thresh:.4f})")
        print(f"  Segment Accuracy       : {acc:.2f} %")
        print(f"  Segment F1-Score       : {f1:.2f} %")
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
            print(f"Results saved to: {result_file}")

        self.test_loss.reset()
        self.test_eer.reset()
        self.test_f1_acc.reset()

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