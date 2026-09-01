import math
import os
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. SSL Audio Encoder (Frontend)
# ==============================================================================

class BaseSSLEncoder(nn.Module):
    """Abstract Base Class for Self-Supervised Learning (SSL) Audio Encoders."""

    def __init__(self, hid_dim: int = 1024, freeze_ssl: bool = True) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.freeze_ssl = freeze_ssl

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class WavLMEncoder(BaseSSLEncoder):
    """
    WavLM Audio Encoder supporting local s3prl checkpoints and HuggingFace models.
    """

    def __init__(
        self,
        ckpt: Optional[str] = None,
        mode: str = "s3prl",
        hid_dim: int = 1024,
        freeze_ssl: bool = True,
    ) -> None:
        super().__init__(hid_dim=hid_dim, freeze_ssl=freeze_ssl)
        self.mode = mode.lower()

        if self.mode in ("s3prl", "s3prl_weighted"):
            if ckpt is None or not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint not found for s3prl mode: {ckpt}")
            
            import contextlib
            import warnings
            with warnings.catch_warnings(), contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                warnings.filterwarnings("ignore")
                import s3prl.hub as hub
                self.ssl_model = hub.wavlm_local(ckpt=ckpt)

        elif self.mode in ("hf", "huggingface"):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                from transformers import WavLMModel
                model_name_or_path = ckpt if ckpt is not None else "microsoft/wavlm-large"
                self.ssl_model = WavLMModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Unsupported SSL mode: {mode}")

        if freeze_ssl:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Audio waveform tensor [Batch, Samples] or [Batch, Samples, 1].
        Returns:
            torch.Tensor: Feature tensor [Batch, Frames, HiddenDim] (~50 frames/s).
        """
        if x.ndim == 3:
            x = x[:, :, 0]

        # Pad to avoid boundary edge distortions
        x = F.pad(x, (0, 256), mode="constant", value=0.0)

        if self.mode == "s3prl":
            if self.freeze_ssl:
                with torch.no_grad():
                    out = self.ssl_model(x)["hidden_states"][-1]
            else:
                out = self.ssl_model(x)["hidden_states"][-1]
            return out
        else:
            if self.freeze_ssl:
                with torch.no_grad():
                    out = self.ssl_model(x).last_hidden_state
            else:
                out = self.ssl_model(x).last_hidden_state
            return out


# ==============================================================================
# 2. Conformer Modules (Backend)
# ==============================================================================

class Swish(nn.Module):
    """Swish / SiLU activation: x * sigmoid(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding for Conformer Self-Attention."""

    def __init__(self, d_model: int, maxlen: int = 1000) -> None:
        super().__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(2 * maxlen, d_model)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        pos_seq = pos_seq.clamp(-self.maxlen, self.maxlen - 1) + self.maxlen
        return self.pe_k(pos_seq)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with optional Relative Positional Encoding."""

    def __init__(self, n_units: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, batch_size: int, pos_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        att_score = torch.matmul(q, k.transpose(-2, -1))

        if pos_k is not None:
            att_score_pos = torch.matmul(q.reshape(batch_size * self.h, -1, self.d_k).transpose(0, 1), pos_k.transpose(-2, -1))
            att_score_pos = att_score_pos.transpose(0, 1).reshape(batch_size, self.h, pos_k.size(0), pos_k.size(1))
            scores = (att_score + att_score_pos) / math.sqrt(self.d_k)
        else:
            scores = att_score / math.sqrt(self.d_k)

        att = F.softmax(scores, dim=-1)
        p_att = self.dropout(att)
        out = torch.matmul(p_att, v).permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(out)


class ConformerMHA(nn.Module):
    """Conformer Multi-Head Self-Attention wrapper with LayerNorm, Dropout, Residual."""

    def __init__(self, in_size: int = 1024, num_head: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.mha = MultiHeadSelfAttention(n_units=in_size, h=num_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, time, idim = x.shape
        x_flat = x.reshape(-1, idim)
        normed = self.ln_norm(x_flat)
        out = self.mha(normed, bs, pos_k)
        out = self.dropout(out)
        return (x_flat + out).reshape(bs, time, idim)


class PositionwiseFeedForward(nn.Module):
    """Conformer Macaron-style Position-wise Feed Forward Layer."""

    def __init__(self, in_size: int = 1024, ffn_hidden: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.w_1 = nn.Linear(in_size, ffn_hidden)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_hidden, in_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ln_norm(x)
        x = self.w_1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)
        return res + 0.5 * x


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module:
    LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(self, channels: int = 1024, kernel_size: int = 31, dropout_rate: float = 0.1) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd"
        self.ln_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.bn_norm = nn.BatchNorm1d(channels)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ln_norm(x).transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.bn_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x).transpose(1, 2)  # [B, T, C]
        return res + x


class ConformerBlock(nn.Module):
    """
    Single Conformer Block with Macaron-style Feed-Forward, MHA, and Convolution Module.
    """

    def __init__(
        self,
        in_size: int = 1024,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(in_size, ffn_hidden, dropout)
        self.mha = ConformerMHA(in_size, num_head, dropout)
        self.conv = ConvolutionModule(in_size, kernel_size, dropout)
        self.ffn2 = PositionwiseFeedForward(in_size, ffn_hidden, dropout)
        self.final_ln = nn.LayerNorm(in_size)

    def forward(self, x: torch.Tensor, pos_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mha(x, pos_k)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_ln(x)


class ConformerEncoder(nn.Module):
    """Conformer Backend composed of stacked Conformer blocks."""

    def __init__(
        self,
        attention_in: int = 1024,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
    ) -> None:
        super().__init__()
        self.use_posi = use_posi
        self.pos_emb = RelativePositionalEncoding(attention_in // num_head) if use_posi else None

        self.blocks = nn.ModuleList([
            ConformerBlock(
                in_size=attention_in,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layer)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_k = None
        if self.use_posi and self.pos_emb is not None:
            t = x.size(1)
            pos = torch.arange(t, device=x.device)
            pos_seq = pos.unsqueeze(0) - pos.unsqueeze(1)
            pos_k = self.pos_emb(pos_seq)

        for block in self.blocks:
            x = block(x, pos_k)
        return x


# ==============================================================================
# 3. Pooling Heads
# ==============================================================================

class SelfWeightedPooling(nn.Module):
    """Self-Weighted Attention Pooling for temporal feature aggregation."""

    def __init__(self, feature_dim: int, num_head: int = 1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head
        self.mm_weights = nn.Parameter(torch.Tensor(feature_dim, num_head))
        nn.init.kaiming_uniform_(self.mm_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, t, d = inputs.shape
        weights = torch.bmm(inputs, self.mm_weights.unsqueeze(0).repeat(b, 1, 1))
        attentions = F.softmax(torch.tanh(weights), dim=1)

        if self.num_head == 1:
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            weighted = torch.bmm(inputs.reshape(-1, d, 1), attentions.view(-1, 1, self.num_head))
            weighted = weighted.view(b, -1, d * self.num_head)

        return weighted.sum(1)


class PoolHead(nn.Module):
    """Downsamples frame-level features (~50 fps) into segment representations."""

    def __init__(
        self,
        hid_dim: int,
        pool: str = "att",
        resolution_train: float = 0.16,
        resolution_test: float = 0.16,
        num_head: int = 1,
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.scale_train = int(resolution_train // 0.02)
        self.scale_test = int(resolution_test // 0.02)
        self.pool = pool
        self.num_head = num_head

        if self.pool == "att":
            self.att_pool = SelfWeightedPooling(self.hid_dim, num_head=num_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale_train if self.training else self.scale_test
        b, f, d = x.shape

        usable_frames = (f // scale) * scale
        if usable_frames < f:
            x = x[:, :usable_frames, :]
            f = usable_frames

        if self.pool == "avg":
            x = x.transpose(1, 2)
            x = F.avg_pool1d(x, kernel_size=scale, stride=scale).transpose(1, 2)
        else:
            x = x.reshape(-1, scale, d)
            x = self.att_pool(x)
            x = x.reshape(b, -1, d * self.num_head)
        return x


# ==============================================================================
# 4. End-to-End Architecture
# ==============================================================================

class WavLMConformer(nn.Module):
    """
    End-to-End WavLM-Conformer Model Architecture.
    Pipeline: Raw Waveform -> WavLM Encoder -> Conformer Backend -> Pool Head -> Classifier
    """

    def __init__(
        self,
        ssl_encoder: str = "wavlm_large",
        ssl_ckpt: Optional[str] = None,
        ssl_mode: str = "s3prl",
        hid_dim: int = 1024,
        freeze_ssl: bool = True,
        conformer_layers: int = 2,
        conformer_heads: int = 4,
        conformer_kernel_size: int = 31,
        conformer_ffn_hidden: int = 1024,
        conformer_dropout: float = 0.1,
        use_relative_pos: bool = False,
        resolution_train: float = 0.16,
        resolution_test: float = 0.16,
        pool: str = "att",
        pool_heads: int = 1,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # 1. SSL Audio Encoder
        self.ssl_encoder = WavLMEncoder(
            ckpt=ssl_ckpt,
            mode=ssl_mode,
            hid_dim=hid_dim,
            freeze_ssl=freeze_ssl,
        )

        # 2. Conformer Backend
        self.conformer = ConformerEncoder(
            attention_in=hid_dim,
            ffn_hidden=conformer_ffn_hidden,
            num_head=conformer_heads,
            num_layer=conformer_layers,
            kernel_size=conformer_kernel_size,
            dropout=conformer_dropout,
            use_posi=use_relative_pos,
        )

        # 3. Pooling Head
        self.pool_head = PoolHead(
            hid_dim=hid_dim,
            pool=pool,
            resolution_train=resolution_train,
            resolution_test=resolution_test,
            num_head=pool_heads,
        )
        self.emb_dim = hid_dim * pool_heads

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.emb_dim, 256),
            nn.SELU(),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent segment embeddings [Batch, T_segments, emb_dim]."""
        ssl_feats = self.ssl_encoder(x)
        conformer_feats = self.conformer(ssl_feats)
        return self.pool_head(conformer_feats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - logits: [Batch, T_segments, 2] (0 = fake, 1 = real)
            - embeddings: [Batch, T_segments, emb_dim] (for Contrastive Loss)
        """
        embeddings = self.forward_features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings
