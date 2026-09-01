from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


# ==============================================================================
# 1. Loss Functions
# ==============================================================================

class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with dynamic padding mask support.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.dim() == 3 and logits.size(-1) == 2:
            logits = logits.transpose(1, 2)  # [B, 2, T]

        loss = self.criterion(logits, targets.long())

        if mask is not None:
            mask_float = mask.float()
            total_valid = mask_float.sum()
            return (loss * mask_float).sum() / total_valid if total_valid > 0 else loss.mean()
        return loss.mean()


class ContrastiveSegmentLoss(nn.Module):
    """
    Contrastive Segment Loss for Speech Editing Detection:
    - Intra-class cohesion: pulls segments of the same class (real/fake) to their centroid.
    - Inter-class separation: pushes real and fake centroids apart by margin m.
    """

    def __init__(
        self,
        margin: float = 1.0,
        alpha_intra: float = 1.0,
        beta_inter: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.alpha_intra = alpha_intra
        self.beta_inter = beta_inter
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = embeddings.size(0)
        intra_losses, inter_losses = [], []

        for b in range(batch_size):
            if mask is not None:
                valid_mask = mask[b].bool()
                feat_b = embeddings[b][valid_mask]
                label_b = labels[b][valid_mask].long()
            else:
                feat_b = embeddings[b]
                label_b = labels[b].long()

            if feat_b.size(0) == 0:
                continue

            real_mask = label_b == 1
            fake_mask = label_b == 0
            num_real, num_fake = real_mask.sum().item(), fake_mask.sum().item()

            c_real, c_fake = None, None
            intra_b_components = []

            # Intra cohesion for real
            if num_real > 0:
                feat_real = feat_b[real_mask]
                c_real = feat_real.mean(dim=0, keepdim=True)
                cos_real = F.cosine_similarity(feat_real, c_real, dim=-1, eps=self.eps)
                intra_b_components.append((1.0 - cos_real).mean())

            # Intra cohesion for fake
            if num_fake > 0:
                feat_fake = feat_b[fake_mask]
                c_fake = feat_fake.mean(dim=0, keepdim=True)
                cos_fake = F.cosine_similarity(feat_fake, c_fake, dim=-1, eps=self.eps)
                intra_b_components.append((1.0 - cos_fake).mean())

            if intra_b_components:
                intra_losses.append(torch.stack(intra_b_components).mean())

            # Inter margin separation
            if num_real > 0 and num_fake > 0 and c_real is not None and c_fake is not None:
                cos_centroids = F.cosine_similarity(c_real, c_fake, dim=-1, eps=self.eps).squeeze()
                dist_centroids = 1.0 - cos_centroids
                inter_losses.append(F.relu(self.margin - dist_centroids))

        device = embeddings.device
        intra_loss = torch.stack(intra_losses).mean() if intra_losses else torch.tensor(0.0, device=device)
        inter_loss = torch.stack(inter_losses).mean() if inter_losses else torch.tensor(0.0, device=device)

        total_loss = self.alpha_intra * intra_loss + self.beta_inter * inter_loss
        stats = {
            "loss_intra": intra_loss.detach(),
            "loss_inter": inter_loss.detach(),
        }
        return total_loss, stats


class TotalLoss(nn.Module):
    """
    Combined Loss: L_Total = L_BCE + lambda_contrastive * L_Contrastive
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.5,
        margin: float = 1.0,
        alpha_intra: float = 1.0,
        beta_inter: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.bce_loss = MaskedCrossEntropyLoss(weight=class_weights)
        self.contrastive_loss = ContrastiveSegmentLoss(
            margin=margin, alpha_intra=alpha_intra, beta_inter=beta_inter
        )

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_ce = self.bce_loss(logits, targets, mask)
        loss_cont, cont_stats = self.contrastive_loss(embeddings, targets, mask)
        total_loss = loss_ce + self.lambda_contrastive * loss_cont

        loss_dict = {
            "loss_total": total_loss,
            "loss_bce": loss_ce.detach(),
            "loss_contrastive": loss_cont.detach(),
            "loss_intra": cont_stats["loss_intra"],
            "loss_inter": cont_stats["loss_inter"],
        }
        return total_loss, loss_dict


# ==============================================================================
# 2. Evaluation Metrics
# ==============================================================================

class EERMetric(Metric):
    """Equal Error Rate (EER) Metric for segment-level detection."""

    def __init__(self, percent: bool = True) -> None:
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())

    def compute(self) -> Tuple[float, float]:
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]

        if not self.y_pred or not self.y_true:
            return 0.0, 0.5

        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)

        if y_pred.numel() == 0:
            return 0.0, 0.5

        sorted_indices = torch.argsort(y_pred, descending=True)
        y_true_sorted = y_true[sorted_indices]

        tp = torch.cumsum(y_true_sorted, dim=0)
        fp = torch.cumsum(1 - y_true_sorted, dim=0)

        pos_total = tp[-1]
        neg_total = fp[-1]

        if pos_total == 0 or neg_total == 0:
            return (100.0 if self.percent else 1.0), 0.0

        tpr = tp / pos_total
        fpr = fp / neg_total

        abs_diff = torch.abs(fpr - (1 - tpr))
        eer_index = torch.argmin(abs_diff)
        eer = fpr[eer_index].item()
        thresh = y_pred[sorted_indices][eer_index].item()

        if self.percent:
            eer *= 100.0
        return eer, thresh


class F1Metric(Metric):
    """Accuracy and F1 Score Metric for binary segment classification."""

    def __init__(self, percent: bool = True) -> None:
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())

    def compute(self) -> Tuple[float, float]:
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]

        if not self.y_pred or not self.y_true:
            return 0.0, 0.0

        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)

        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=-1)

        correct = (y_pred == y_true).sum().float()
        total = y_true.size(0)
        acc = (correct / total).item() if total > 0 else 0.0

        tp = (y_pred * y_true).sum().float()
        fp = (y_pred * (1 - y_true)).sum().float()
        fn = ((1 - y_pred) * y_true).sum().float()

        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if self.percent:
            acc *= 100.0
            f1 *= 100.0
        return acc, f1
