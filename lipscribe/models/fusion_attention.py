"""Cross-modal attention, gated fusion, and proposed LipScribe model P (W4-01, W4-02).

Implements Section 7.3--7.5 (Algorithm 1) of the Technical Implementation Document
and Section 3 of ``docs/INTERFACES.md``:

1. **Stream projections (W4-01):**
   Each 512-d encoder sequence is projected to ``d = 256`` with LayerNorm:
   ``A = LN(H_a W_a)``, ``V = LN(H_v W_v)``. If temporal lengths differ by
   1--2 frames at clip boundaries, the longer sequence is truncated to the shorter.

2. **Missing-modality vector and modality dropout (W4-02):**
   A single learned vector ``e_miss`` in ``R^256`` replaces a dropped stream's
   projected encoding across all frames (never an attention mask, which would
   mask all keys and produce NaN).
   * In ``model.train()``, per utterance ``u ~ U(0, 1)`` drops audio when
     ``u < p_audio`` (default 0.25) and drops video when
     ``p_audio <= u < p_audio + p_visual`` (default 0.15); never both.
   * At inference (``model.eval()``), modality dropout is disabled. If a face is
     missing in ``> 50 %`` of the window's frames (via ``video_valid``), the
     visual stream is replaced by ``e_miss``. Explicit test-time ablations
     (``drop_modality="audio"`` or ``"visual"``) likewise substitute ``e_miss``.

3. **Bidirectional cross-modal multi-head attention (W4-01):**
   Two 4-head ``nn.MultiheadAttention`` blocks (``d = 256``, ``dropout = 0.1``,
   ``batch_first = True``) with residual connections and LayerNorm:
   ``A_tilde = LN(A + MHA_{A->V}(A, V, V))`` and
   ``V_tilde = LN(V + MHA_{V->A}(V, A, A))``.

4. **Per-frame gated fusion (W4-02):**
   ``g_t = sigma(w_g^T [a_tilde_t ; v_tilde_t] + b_g) in (0, 1)``,
   ``f_t = g_t * a_tilde_t + (1 - g_t) * v_tilde_t``.
"""
from __future__ import annotations

import torch
from torch import nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import build_visual_encoder

AUDIO_FEATURE_DIMS = {
    "logmel80": 80,
    "mfcc39": 39,
    "logmel_mfcc": 119,
}


def _make_key_padding_mask(mask: torch.Tensor | None, batch_size: int, seq_len: int,
                           device: torch.device) -> torch.Tensor | None:
    """Convert LipScribe's ``mask`` (True = real frame, False = padding) into
    PyTorch MultiheadAttention's ``key_padding_mask`` (True = ignore / padded).

    If every frame of an item is marked as padding, frame 0 is unmasked so
    softmax over keys remains finite and never produces NaN.
    """
    if mask is None:
        return None
    valid = mask.to(device=device, dtype=torch.bool)[:, :seq_len]
    if valid.size(1) < seq_len:
        pad = torch.zeros(batch_size, seq_len - valid.size(1), dtype=torch.bool, device=device)
        valid = torch.cat([valid, pad], dim=1)
    key_padding_mask = ~valid
    all_masked = key_padding_mask.all(dim=-1)
    if all_masked.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, 0] = False
    return key_padding_mask


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal multi-head attention (Card W4-01, Section 7.3).

    Projects 512-d audio and visual encoder outputs to ``embed_dim = 256`` with
    LayerNorm, truncates the longer sequence if lengths differ by 1--2 frames,
    and applies two 4-head MHA blocks (audio->video and video->audio) with
    residual connections and LayerNorm.
    """

    def __init__(
        self,
        in_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.proj_a = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.mha_a2v = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.mha_v2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

    def project(self, a: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate to common frame length ``T = min(T_a, T_v)`` and project to ``embed_dim``."""
        if a.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"CrossModalAttention expects 3D tensors [B, T, D], got {tuple(a.shape)} and {tuple(v.shape)}"
            )
        t = min(a.size(1), v.size(1))
        a = a[:, :t, :]
        v = v[:, :t, :]

        a_proj = self.proj_a(a) if a.size(-1) == self.in_dim else a
        v_proj = self.proj_v(v) if v.size(-1) == self.in_dim else v
        return a_proj, v_proj

    def attend(
        self,
        a_proj: torch.Tensor,
        v_proj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run A->V and V->A multi-head attention on projected `[B, T, 256]` streams."""
        t = min(a_proj.size(1), v_proj.size(1))
        a_proj = a_proj[:, :t, :]
        v_proj = v_proj[:, :t, :]

        key_padding_mask = _make_key_padding_mask(mask, a_proj.size(0), t, a_proj.device)

        # Audio queries attend to video keys/values (A -> V)
        attn_a, weights_a2v = self.mha_a2v(
            query=a_proj,
            key=v_proj,
            value=v_proj,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        a_tilde = self.norm_a(a_proj + attn_a)

        # Video queries attend to audio keys/values (V -> A)
        attn_v, weights_v2a = self.mha_v2a(
            query=v_proj,
            key=a_proj,
            value=a_proj,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        v_tilde = self.norm_v(v_proj + attn_v)

        return a_tilde, v_tilde, {"a2v": weights_a2v, "v2a": weights_v2a}

    def forward(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Project and cross-attend audio and visual streams.

        Args:
            a: ``[B, T_a, 512]`` (or already projected ``[B, T_a, 256]``)
            v: ``[B, T_v, 512]`` (or already projected ``[B, T_v, 256]``)
            mask: optional ``bool[B, T]`` with ``True`` on real frames, ``False`` on padding
            return_weights: if True, also return ``{"a2v": [B, T, T], "v2a": [B, T, T]}``

        Returns:
            ``(a_tilde, v_tilde)`` each ``[B, T, 256]``, plus attention weights if requested.
        """
        a_proj, v_proj = self.project(a, v)
        a_tilde, v_tilde, weights = self.attend(a_proj, v_proj, mask=mask)
        if return_weights:
            return a_tilde, v_tilde, weights
        return a_tilde, v_tilde


class GatedFusion(nn.Module):
    """Per-frame scalar gate ``g_t = sigma(w_g^T [a_tilde_t; v_tilde_t] + b_g)`` (Card W4-02).

    Returns:
        fused: ``[B, T, 256]`` where ``f_t = g_t * a_tilde_t + (1 - g_t) * v_tilde_t``
        gate: ``[B, T]`` in ``(0, 1)`` (near 1 = relies on audio, near 0 = relies on lips)
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(2 * embed_dim, 1)

    def forward(
        self,
        a_tilde: torch.Tensor,
        v_tilde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([a_tilde, v_tilde], dim=-1)  # [B, T, 512]
        g = torch.sigmoid(self.gate_proj(concat))       # [B, T, 1]
        fused = g * a_tilde + (1.0 - g) * v_tilde       # [B, T, 256]
        return fused, g.squeeze(-1)


class AttentionFusion(nn.Module):
    """Complete cross-modal attention and gated fusion with ``e_miss`` and modality dropout (W4-01, W4-02).

    Matches ``docs/INTERFACES.md`` Section 3:
        ``AttentionFusion(a, v, mask) -> fused: [B, T, 256], gate: [B, T]``
    """

    def __init__(
        self,
        in_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        p_audio_drop: float = 0.25,
        p_visual_drop: float = 0.15,
    ) -> None:
        super().__init__()
        if p_audio_drop < 0.0 or p_visual_drop < 0.0 or (p_audio_drop + p_visual_drop) > 1.0:
            raise ValueError(
                f"Modality dropout probabilities must be >= 0 and sum to <= 1.0, "
                f"got p_audio_drop={p_audio_drop}, p_visual_drop={p_visual_drop}"
            )
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.p_audio_drop = float(p_audio_drop)
        self.p_visual_drop = float(p_visual_drop)

        # Learned 256-d missing-modality vector (Algorithm 1, Section 7.4).
        self.e_miss = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.e_miss, mean=0.0, std=0.02)

        self.cross_attn = CrossModalAttention(
            in_dim=in_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.gate = GatedFusion(embed_dim=embed_dim)

    def sample_modality_dropout(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw disjoint per-utterance boolean drop masks ``(drop_audio, drop_visual)``.

        Only active when ``self.training`` is True; never drops both streams for
        the same utterance.
        """
        if not self.training or (self.p_audio_drop == 0.0 and self.p_visual_drop == 0.0):
            zeros = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return zeros, zeros
        u = torch.rand(batch_size, device=device)
        drop_a = u < self.p_audio_drop
        drop_v = (u >= self.p_audio_drop) & (u < self.p_audio_drop + self.p_visual_drop)
        return drop_a, drop_v

    @staticmethod
    def detect_missing_face(
        video_valid: torch.Tensor | None,
        seq_len: int,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return ``bool[B]`` True where a face is missing in > 50 % of the window's frames."""
        if video_valid is None:
            return None
        device = video_valid.device
        valid = video_valid.to(dtype=torch.bool)[:, :seq_len]
        B, t = valid.shape

        if lengths is not None:
            frame_mask = (
                torch.arange(t, device=device).unsqueeze(0)
                < lengths.to(device=device, dtype=torch.long).clamp(min=1, max=t).unsqueeze(1)
            )
        elif mask is not None:
            frame_mask = mask.to(device=device, dtype=torch.bool)[:, :t]
        else:
            frame_mask = torch.ones(B, t, dtype=torch.bool, device=device)

        total_frames = frame_mask.sum(dim=1).clamp(min=1).float()
        valid_frames = (valid & frame_mask).sum(dim=1).float()
        missing_fraction = 1.0 - (valid_frames / total_frames)
        return missing_fraction > 0.5

    def forward(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        video_valid: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        drop_modality: str | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse audio and visual encodings into ``[B, T, 256]`` and per-frame gate ``[B, T]``.

        Args:
            a: audio encoder sequence ``[B, T_a, 512]``
            v: visual encoder sequence ``[B, T_v, 512]``
            mask: optional padding mask ``bool[B, T]`` (True on real frames, False on padding)
            video_valid: optional per-frame Face Mesh validity ``bool[B, T]``
            lengths: optional true frame counts ``int64[B]``
            drop_modality: optional ``"audio"`` or ``"visual"`` for test-time ablations
            return_weights: if True, also return ``{"a2v": [B, T, T], "v2a": [B, T, T]}``
        """
        a_proj, v_proj = self.cross_attn.project(a, v)
        B, t, _ = a_proj.shape
        device = a_proj.device

        if mask is None and lengths is not None:
            mask = (
                torch.arange(t, device=device).unsqueeze(0)
                < lengths.to(device=device, dtype=torch.long).clamp(min=1, max=t).unsqueeze(1)
            )
        elif mask is not None:
            mask = mask[:, :t]

        drop_a, drop_v = self.sample_modality_dropout(B, device)

        # Inference / missing-face fallback: if > 50 % of the window's frames have no face,
        # replace the visual stream with e_miss (without ever dropping both streams at once).
        face_missing = self.detect_missing_face(video_valid, t, mask=mask, lengths=lengths)
        if face_missing is not None:
            drop_v = drop_v | (face_missing.to(device=device) & ~drop_a)

        # Explicit test-time modality ablation (W3-02, W5-06)
        if drop_modality == "audio":
            drop_a = torch.ones(B, dtype=torch.bool, device=device)
            drop_v = torch.zeros(B, dtype=torch.bool, device=device)
        elif drop_modality == "visual":
            drop_v = torch.ones(B, dtype=torch.bool, device=device)
            drop_a = torch.zeros(B, dtype=torch.bool, device=device)
        elif drop_modality is not None:
            raise ValueError(f"Unsupported drop_modality: {drop_modality!r}")

        e_miss_a = self.e_miss.to(dtype=a_proj.dtype).view(1, 1, -1).expand(B, t, -1)
        e_miss_v = self.e_miss.to(dtype=v_proj.dtype).view(1, 1, -1).expand(B, t, -1)
        if drop_a.any():
            a_proj = torch.where(drop_a.view(B, 1, 1), e_miss_a, a_proj)
        if drop_v.any():
            v_proj = torch.where(drop_v.view(B, 1, 1), e_miss_v, v_proj)

        a_tilde, v_tilde, weights = self.cross_attn.attend(a_proj, v_proj, mask=mask)
        fused, gate = self.gate(a_tilde, v_tilde)

        if return_weights:
            return fused, gate, weights
        return fused, gate


class LipScribeModel(nn.Module):
    """Proposed audio-visual model P (Section 7, Algorithm 1, ``docs/INTERFACES.md``).

    Architecture:
        AudioEncoder (512) + VisualEncoder (512)
        -> AttentionFusion (Projections 256, e_miss modality dropout, 4-head A->V & V->A MHA, Gate)
        -> CTCHead (1-layer BiLSTM 128/dir -> Linear 256->28 -> log-softmax)
    """

    def __init__(
        self,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        num_heads: int = 4,
        p_audio_drop: float = 0.25,
        p_visual_drop: float = 0.15,
        model_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        cfg = model_cfg or {}
        audio_feat_name = cfg.get("audio_features", "logmel80")
        in_dim = int(cfg.get("audio_in_dim", AUDIO_FEATURE_DIMS.get(audio_feat_name, 80)))

        self.audio_encoder = AudioEncoder(in_dim=in_dim, dropout=dropout)
        self.visual_encoder = build_visual_encoder(cfg, dropout=dropout)
        self.fusion = AttentionFusion(
            in_dim=512,
            embed_dim=256,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            p_audio_drop=p_audio_drop,
            p_visual_drop=p_visual_drop,
        )
        self.ctc_head = CTCHead(input_dim=256)

    @property
    def e_miss(self) -> nn.Parameter:
        return self.fusion.e_miss

    @property
    def p_audio_drop(self) -> float:
        return self.fusion.p_audio_drop

    @property
    def p_visual_drop(self) -> float:
        return self.fusion.p_visual_drop

    def forward(self, batch: dict) -> dict:
        audio_feat, a_lens = self.audio_encoder(batch["features"], batch.get("feat_lengths"))
        v_lens = batch.get("lengths")
        video_feat = self.visual_encoder(batch["video"], v_lens)

        t = min(audio_feat.size(1), video_feat.size(1))
        audio_feat = audio_feat[:, :t, :]
        video_feat = video_feat[:, :t, :]

        if a_lens is not None and v_lens is not None:
            lengths = torch.minimum(
                a_lens.to(device=audio_feat.device, dtype=torch.long),
                v_lens.to(device=audio_feat.device, dtype=torch.long),
            ).clamp(min=1, max=t)
        elif a_lens is not None:
            lengths = a_lens.to(device=audio_feat.device, dtype=torch.long).clamp(min=1, max=t)
        elif v_lens is not None:
            lengths = v_lens.to(device=audio_feat.device, dtype=torch.long).clamp(min=1, max=t)
        else:
            lengths = torch.full((audio_feat.size(0),), t, device=audio_feat.device, dtype=torch.long)

        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device=audio_feat.device, dtype=torch.bool)[:, :t]
        else:
            mask = torch.arange(t, device=audio_feat.device).unsqueeze(0) < lengths.unsqueeze(1)

        video_valid = batch.get("video_valid")
        if video_valid is not None:
            video_valid = video_valid.to(device=audio_feat.device, dtype=torch.bool)[:, :t]

        fused, gate, attn_weights = self.fusion(
            audio_feat,
            video_feat,
            mask=mask,
            video_valid=video_valid,
            lengths=lengths,
            drop_modality=batch.get("drop_modality"),
            return_weights=True,
        )
        log_probs = self.ctc_head(fused, lengths)
        return {
            "log_probs": log_probs,
            "lengths": lengths,
            "gate": gate,
            "attn_a2v": attn_weights["a2v"],
            "attn_v2a": attn_weights["v2a"],
        }
