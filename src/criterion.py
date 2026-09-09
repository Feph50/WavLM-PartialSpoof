from typing import Optional, Dict, Tuple, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment


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
        if logits.dim() == 3 and targets.dim() == 2:
            logits = logits.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        num_classes = logits.size(1)
        safe_targets = targets.long().clone()
        invalid = (safe_targets < 0) | (safe_targets >= num_classes)
        safe_targets[invalid] = 0

        loss = self.criterion(logits, safe_targets)

        if mask is not None:
            effective_mask = mask.float() * (~invalid).float()
            total_valid = effective_mask.sum()
            if total_valid > 0:
                return (loss * effective_mask).sum() / total_valid
            else:
                return (loss * 0.0).sum()
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


class JIBonaMetric(Metric):
    """
    Jaccard Index Error (JI_bona) for Bona Fide Detection (Zhang et al. 2024, Koo et al. 2025).
    Eq. (1) & (3) in Interspeech 2024:
      JI_bona,j = (FA_bona,j + MD_bona,j) / TOTAL_bona,j
      JI_bona = (1 / |D|) * sum_{j in D} JI_bona,j
    Applies Oracle VAD by excluding non-speech frames (labels: 0, 101, 102..120).
    """

    def __init__(
        self,
        percent: bool = True,
        bonafide_pred_label: int = -1,
        bonafide_gt_label: int = 1,
        oracle_vad: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if "bonafide_label" in kwargs:
            bonafide_pred_label = kwargs["bonafide_label"]
        self.percent = percent
        self.bonafide_pred_label = bonafide_pred_label
        self.bonafide_gt_label = bonafide_gt_label
        self.oracle_vad = oracle_vad
        self.add_state("total_ji", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_utterances", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = preds.size(0)
        device = preds.device

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        lengths_np = label_lengths.detach().cpu().numpy() if label_lengths is not None else None

        for b in range(batch_size):
            max_avail = min(preds_np.shape[1], targets_np.shape[1])
            valid_len = min(int(lengths_np[b]), max_avail) if lengths_np is not None else max_avail
            p_seq = preds_np[b, :valid_len]
            t_seq = targets_np[b, :valid_len]

            # Oracle VAD: filter out nonspeech frames (0, 101, 102..120)
            if self.oracle_vad:
                speech_mask = (t_seq != 0) & (t_seq < 101)
                if not np.any(speech_mask):
                    continue
                p_seq = p_seq[speech_mask]
                t_seq = t_seq[speech_mask]

            ref_bona = (t_seq == self.bonafide_gt_label)
            hyp_bona = (p_seq == self.bonafide_pred_label)

            inter = np.logical_and(ref_bona, hyp_bona).sum()
            union = np.logical_or(ref_bona, hyp_bona).sum()

            if union == 0:
                utt_ji = 0.0
            else:
                fa = np.logical_and(hyp_bona, ~ref_bona).sum()
                md = np.logical_and(ref_bona, ~hyp_bona).sum()
                utt_ji = float(fa + md) / float(union)

            self.total_ji += torch.tensor(utt_ji, device=device)
            self.total_utterances += torch.tensor(1, device=device)

    def compute(self) -> float:
        if self.total_utterances.item() == 0:
            return 0.0
        avg_ji = (self.total_ji / self.total_utterances.float()).item()
        if self.percent:
            avg_ji *= 100.0
        return avg_ji


class JERMetric(Metric):
    """
    Jaccard Error Rate (JER_spoof) Metric for Spoof Diarization (Zhang et al. 2024, Koo et al. 2025).
    Eq. (2) & (4) in Interspeech 2024:
      JER_spoof,j = (1 / |Aj|) * sum_{Ai in Aj} (FA_Ai,j + MD_Ai,j) / TOTAL_Ai,j
      JER_global_spoof = (1 / sum |Aj|) * sum_j sum_{Ai in Aj} JER_Ai,j
    Computes optimal bipartite matching between predicted spoof labels/clusters and reference labels
    using the Hungarian algorithm (linear_sum_assignment) per utterance.
    Applies Oracle VAD by excluding non-speech frames (labels: 0, 101, 102..120).
    """

    def __init__(
        self,
        percent: bool = True,
        bonafide_pred_label: int = -1,
        bonafide_gt_label: int = 1,
        oracle_vad: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if "bonafide_label" in kwargs:
            bonafide_pred_label = kwargs["bonafide_label"]
        self.percent = percent
        self.bonafide_pred_label = bonafide_pred_label
        self.bonafide_gt_label = bonafide_gt_label
        self.oracle_vad = oracle_vad
        self.add_state("total_spoof_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_attacks", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            preds: [B, T] predicted labels / cluster IDs (with bonafide_pred_label for genuine frames).
            targets: [B, T] ground truth labels.
            label_lengths: Optional [B] valid lengths for each sample in batch.
        """
        batch_size = preds.size(0)
        device = preds.device

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        lengths_np = label_lengths.detach().cpu().numpy() if label_lengths is not None else None

        for b in range(batch_size):
            max_avail = min(preds_np.shape[1], targets_np.shape[1])
            valid_len = min(int(lengths_np[b]), max_avail) if lengths_np is not None else max_avail
            p_seq = preds_np[b, :valid_len]
            t_seq = targets_np[b, :valid_len]

            # Oracle VAD: filter out nonspeech frames (0, 101, 102..120)
            if self.oracle_vad:
                speech_mask = (t_seq != 0) & (t_seq < 101)
                if not np.any(speech_mask):
                    continue
                p_seq = p_seq[speech_mask]
                t_seq = t_seq[speech_mask]

            # Filter distinct spoof classes (excluding bona fide)
            ref_classes = [c for c in np.unique(t_seq) if c != self.bonafide_gt_label]
            hyp_classes = [c for c in np.unique(p_seq) if c != self.bonafide_pred_label]

            # Case 1: Purely genuine speech in reference
            if len(ref_classes) == 0:
                if len(hyp_classes) > 0:
                    # Hyp predicted spoof frames when ground truth is pure bonafide -> 100% false alarm
                    self.total_spoof_error += torch.tensor(1.0, device=device)
                    self.total_attacks += torch.tensor(1, device=device)
                continue

            # Case 2: Reference contains spoof attacks, but hypothesis predicted none
            if len(hyp_classes) == 0:
                # Completely missed all spoof attacks -> 100% error
                self.total_spoof_error += torch.tensor(float(len(ref_classes)), device=device)
                self.total_attacks += torch.tensor(len(ref_classes), device=device)
                continue

            # Build Jaccard cost matrix (1 - IoU) for Hungarian matching
            n_ref, n_hyp = len(ref_classes), len(hyp_classes)
            cost_matrix = np.ones((n_ref, n_hyp), dtype=np.float32)

            for i, r_c in enumerate(ref_classes):
                ref_mask = (t_seq == r_c)
                for j, h_c in enumerate(hyp_classes):
                    hyp_mask = (p_seq == h_c)
                    inter = np.logical_and(ref_mask, hyp_mask).sum()
                    union = np.logical_or(ref_mask, hyp_mask).sum()
                    if union > 0:
                        cost_matrix[i, j] = 1.0 - (inter / float(union))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_pairs = dict(zip(row_ind, col_ind))

            utt_error_sum = 0.0
            # Matched & missed reference classes
            for i, r_c in enumerate(ref_classes):
                ref_mask = (t_seq == r_c)
                if i in matched_pairs:
                    j = matched_pairs[i]
                    h_c = hyp_classes[j]
                    hyp_mask = (p_seq == h_c)
                    fa = np.logical_and(hyp_mask, ~ref_mask).sum()
                    md = np.logical_and(ref_mask, ~hyp_mask).sum()
                    union = np.logical_or(ref_mask, hyp_mask).sum()
                    utt_error_sum += (float(fa + md) / float(union)) if union > 0 else 0.0
                else:
                    # Missed attack class completely
                    utt_error_sum += 1.0

            # Unmatched hypothesis classes (pure false alarm clusters)
            for j, h_c in enumerate(hyp_classes):
                if j not in col_ind:
                    utt_error_sum += 1.0

            self.total_spoof_error += torch.tensor(utt_error_sum, device=device)
            self.total_attacks += torch.tensor(len(ref_classes), device=device)

    def compute(self) -> float:
        if self.total_attacks.item() == 0:
            return 0.0
        avg_jer = (self.total_spoof_error / self.total_attacks.float()).item()
        if self.percent:
            avg_jer *= 100.0
        return avg_jer


# Alias for explicit clarity
JERSpoofMetric = JERMetric


# ==============================================================================
# 3. Combined Diarization Loss
# ==============================================================================

class TotalDiarizationLoss(nn.Module):
    """
    Combined Loss for Spoof Localization and Spoof Diarization:
    L_Total = L_Loc + lambda_dia * L_Dia
    where:
    - L_Loc is TotalLoss (BCE + ContrastiveSegmentLoss) for localization (untouched).
    - L_Dia is MaskedCrossEntropyLoss over spoof attack classes.
    """

    def __init__(
        self,
        loc_loss: Optional[TotalLoss] = None,
        lambda_dia: float = 1.0,
        spoof_only: bool = True,
    ) -> None:
        super().__init__()
        self.loc_loss = loc_loss if loc_loss is not None else TotalLoss()
        self.lambda_dia = lambda_dia
        self.spoof_only = spoof_only
        self.dia_ce_loss = MaskedCrossEntropyLoss()

    def forward(
        self,
        loc_logits: torch.Tensor,
        loc_embeddings: torch.Tensor,
        loc_targets: torch.Tensor,
        dia_logits: torch.Tensor,
        dia_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Localization Loss (original intact)
        loss_loc, loc_dict = self.loc_loss(loc_logits, loc_embeddings, loc_targets, mask=mask)

        # 2. Diarization Loss
        if self.spoof_only:
            # Supervise diarization on spoof frames (loc_targets == 0)
            dia_mask = (loc_targets == 0)
            if mask is not None:
                dia_mask = dia_mask & mask
        else:
            dia_mask = mask

        loss_dia = self.dia_ce_loss(dia_logits, dia_targets, mask=dia_mask)
        total_loss = loss_loc + self.lambda_dia * loss_dia

        loss_dict = {
            **loc_dict,
            "loss_dia": loss_dia.detach(),
            "loss_total_joint": total_loss.detach(),
        }
        return total_loss, loss_dict