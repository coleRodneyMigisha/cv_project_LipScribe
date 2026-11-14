"""Flix v4: transfer-learning encoders and the three v4 models (audio, lips, both).

v3 showed that the GRID-built encoders are too small for open-vocabulary film and TED
speech (see reports/flix_v4/decision_log.md). v4 starts each stream from a pretrained
encoder and fine-tunes it on our film + TED clips; every reported weight is our own.

* ``WavLMAudioEncoder``: WavLM Base+ (microsoft/wavlm-base-plus, self-supervised on 94k h
  of unlabelled speech; no transcripts, so no overlap with our Whisper labels).
  Raw 16 kHz waveform -> ``[B, T50, 768]`` at 50 frames/s (20 ms).
* ``AutoAVSRVisualEncoder``: the visual front-end (3D conv + ResNet-18) and 12-layer
  Conformer of Auto-AVSR's ``vsr_trlrs3vox2_base`` (LRS3 + VoxCeleb2). Its subword
  decoder and CTC layer are dropped. ``[B, 1, T, 96, 96]`` mouth crops in 0-1
  -> ``[B, T, 768]`` at 25 frames/s.

Models (``model.name`` in the config; batches follow docs/INTERFACES.md):

* ``v4_audio``:  WavLM -> dropout -> Linear 768 -> 28 characters (CTC at 50 frames/s)
* ``v4_visual``: Auto-AVSR encoder -> dropout -> Linear 768 -> 28 characters (25 frames/s)
* ``v4_fusion``: both encoders (initialised from the two runs above via ``init_from``),
  video features repeated x2 to 50 frames/s, then exactly Model P's fusion: projections to
  256, bidirectional cross-modal attention, per-frame gate, ``e_miss`` with modality
  dropout, and the BiLSTM CTC head.

All outputs are ``{"log_probs": [T, B, 28], "lengths": [B]}`` like every other LipScribe model.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from lipscribe.data.tokenizer import BLANK_ID, VOCAB_SIZE
from lipscribe.models.ctc_head import CTCHead

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAVLM = "microsoft/wavlm-base-plus"
DEFAULT_AVSR_CKPT = "pretrained/vsr_trlrs3vox2_base.pth"
AVSR_MEAN, AVSR_STD, AVSR_CROP = 0.421, 0.165, 88     # Auto-AVSR datamodule/transforms.py


def _resolve(path: str | os.PathLike | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def frames_mask(lengths: torch.Tensor, t: int) -> torch.Tensor:
    """``bool[B, t]``, True on real frames."""
    return torch.arange(t, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


# --------------------------------------------------------------------------- encoders

class WavLMAudioEncoder(nn.Module):
    """WavLM Base+ on raw audio.

    Args:
        name_or_path: Hugging Face id or local folder (default microsoft/wavlm-base-plus).
        cache_dir: Hugging Face cache folder (relative to the repo root); offline after one download.
        pretrained: False builds the same architecture with random weights (tests, resume).
        mask_time_prob / layerdrop: WavLM's own SpecAugment-style time masking and LayerDrop,
            active in ``train()`` only.
        mask_feature_prob: SpecAugment-style channel masking (spans of 10 channels); 0 = off.
        freeze_feature_encoder: keep the convolutional waveform encoder fixed (standard for
            CTC fine-tuning of wav2vec2-style models; the Transformer is trained).
    """

    out_dim = 768

    def __init__(self, name_or_path: str = DEFAULT_WAVLM, cache_dir: str | None = "pretrained/hf/hub",
                 pretrained: bool = True, mask_time_prob: float = 0.05, layerdrop: float = 0.05,
                 freeze_feature_encoder: bool = True, mask_feature_prob: float = 0.0) -> None:
        super().__init__()
        from transformers import WavLMConfig, WavLMModel

        cache = str(_resolve(cache_dir)) if cache_dir else None
        kwargs = dict(mask_time_prob=mask_time_prob, layerdrop=layerdrop, mask_feature_prob=mask_feature_prob,
                      apply_spec_augment=mask_time_prob > 0 or mask_feature_prob > 0)
        if pretrained:
            self.model = WavLMModel.from_pretrained(name_or_path, cache_dir=cache, **kwargs)
        else:
            cfg = WavLMConfig.from_pretrained(name_or_path, cache_dir=cache, **kwargs)
            self.model = WavLMModel(cfg)
        if freeze_feature_encoder:
            self.model.freeze_feature_encoder()
        self.out_dim = int(self.model.config.hidden_size)

    def output_lengths(self, n_samples: torch.Tensor) -> torch.Tensor:
        return self.model._get_feat_extract_output_lengths(n_samples).to(torch.long)

    def forward(self, wave: torch.Tensor, n_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # wavlm-base-plus: do_normalize=False, return_attention_mask=True (preprocessor_config.json)
        attn = frames_mask(n_samples.to(wave.device), wave.size(1)).long()
        h = self.model(input_values=wave, attention_mask=attn).last_hidden_state
        lengths = self.output_lengths(n_samples.to(wave.device)).clamp(min=1, max=h.size(1))
        return h, lengths


class AutoAVSRVisualEncoder(nn.Module):
    """Auto-AVSR visual front-end + Conformer encoder (vendored in lipscribe.third_party.auto_avsr).

    Input preprocessing matches Auto-AVSR: 88x88 crop of the 96x96 mouth crop (random in
    ``train()``, centre otherwise), then ``(x - 0.421) / 0.165``. Padded frames are set to 0.
    """

    out_dim = 768

    def __init__(self, checkpoint: str | None = DEFAULT_AVSR_CKPT, dropout: float = 0.1,
                 zoom: float = 1.0) -> None:
        super().__init__()
        # Our crops are tighter than Auto-AVSR's (the lips fill most of the 96x96 frame), so
        # ``zoom < 1`` shrinks each frame and pads it back to 96x96 by edge replication, which
        # brings the mouth to roughly the scale the pretrained weights expect.
        self.zoom = float(zoom)
        from lipscribe.third_party.auto_avsr.conformer_encoder import ConformerEncoder
        from lipscribe.third_party.auto_avsr.resnet import video_resnet

        self.frontend = video_resnet()
        self.proj_encoder = nn.Linear(512, 768)
        self.encoder = ConformerEncoder(attention_dim=768, attention_heads=12, linear_units=3072,
                                        num_blocks=12, cnn_module_kernel=31, dropout_rate=dropout,
                                        positional_dropout_rate=dropout)
        path = _resolve(checkpoint)
        if path is not None:
            self.load_avsr(path)

    def load_avsr(self, path: Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        keep = {k: v for k, v in sd.items() if k.split(".")[0] in ("frontend", "proj_encoder", "encoder")}
        own = self.state_dict()
        missing = [k for k in own if k not in keep]
        if missing:
            raise ValueError(f"{path}: not an Auto-AVSR VSR checkpoint (missing {missing[:3]})")
        self.load_state_dict(keep, strict=True)
        print(f"Loaded Auto-AVSR visual encoder from {path} ({len(keep)} tensors)")

    def rescale(self, video: torch.Tensor) -> torch.Tensor:
        if self.zoom == 1.0:
            return video
        B, C, T, H, W = video.shape
        n = max(8, int(round(H * self.zoom)))
        x = nn.functional.interpolate(video.reshape(B * C * T, 1, H, W), size=(n, n),
                                      mode="bilinear", align_corners=False, antialias=True)
        top, left = (H - n) // 2, (W - n) // 2
        x = nn.functional.pad(x, (left, W - n - left, top, H - n - top), mode="replicate")
        return x.reshape(B, C, T, H, W)

    def crop(self, video: torch.Tensor) -> torch.Tensor:
        h, w = video.shape[-2:]
        c = AVSR_CROP
        if self.training:
            # one random offset per clip (the same for all its frames)
            B = video.size(0)
            ys = torch.randint(0, h - c + 1, (B,)).tolist()
            xs = torch.randint(0, w - c + 1, (B,)).tolist()
            return torch.stack([video[b, ..., y:y + c, x:x + c] for b, (y, x) in enumerate(zip(ys, xs))])
        y, x = (h - c) // 2, (w - c) // 2
        return video[..., y:y + c, x:x + c]

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5 or video.size(1) != 1:
            raise ValueError(f"expects [B, 1, T, H, W], got {tuple(video.shape)}")
        B, _, T = video.shape[:3]
        lengths = lengths.to(device=video.device, dtype=torch.long).clamp(min=1, max=T)
        mask = frames_mask(lengths, T)                                   # [B, T]
        x = (self.crop(self.rescale(video)) - AVSR_MEAN) / AVSR_STD
        x = x * mask.view(B, 1, T, 1, 1).to(x.dtype)
        x = self.frontend(x.transpose(1, 2))                             # [B, T, 1, H, W] -> [B, T, 512]
        x = self.proj_encoder(x)
        x, _ = self.encoder(x, mask.unsqueeze(1))                        # mask [B, 1, T]
        return x


# --------------------------------------------------------------------------- models

class LinearCTC(nn.Module):
    """Dropout + Linear -> log-softmax, time-first ``[T, B, V]``."""

    def __init__(self, in_dim: int, dropout: float = 0.1, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, vocab_size)
        with torch.no_grad():
            self.proj.bias[BLANK_ID] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(x)).log_softmax(-1).transpose(0, 1)


def _wavlm_from_cfg(cfg: dict) -> WavLMAudioEncoder:
    a = cfg.get("audio_encoder_cfg") or {}
    return WavLMAudioEncoder(name_or_path=a.get("name", DEFAULT_WAVLM),
                             cache_dir=a.get("cache_dir", "pretrained/hf/hub"),
                             pretrained=bool(a.get("pretrained", True)),
                             mask_time_prob=float(a.get("mask_time_prob", 0.05)),
                             layerdrop=float(a.get("layerdrop", 0.05)),
                             freeze_feature_encoder=bool(a.get("freeze_feature_encoder", True)),
                             mask_feature_prob=float(a.get("mask_feature_prob", 0.0)))


def _avsr_from_cfg(cfg: dict) -> AutoAVSRVisualEncoder:
    v = cfg.get("visual_encoder_cfg") or {}
    ckpt = v.get("checkpoint", DEFAULT_AVSR_CKPT) if v.get("pretrained", True) else None
    return AutoAVSRVisualEncoder(checkpoint=ckpt, dropout=float(v.get("dropout", 0.1)),
                                 zoom=float(v.get("zoom", 1.0)))


class V4AudioOnly(nn.Module):
    def __init__(self, model_cfg: dict | None = None) -> None:
        super().__init__()
        cfg = model_cfg or {}
        self.audio_encoder = _wavlm_from_cfg(cfg)
        self.ctc_head = LinearCTC(self.audio_encoder.out_dim, float(cfg.get("dropout", 0.1)))

    def forward(self, batch: dict) -> dict:
        h, lengths = self.audio_encoder(batch["audio"], batch["audio_lengths"])
        return {"log_probs": self.ctc_head(h), "lengths": lengths}


class V4VisualOnly(nn.Module):
    """``head: avsr_ctc`` keeps Auto-AVSR's own word-piece CTC layer (5,049 outputs, initialised
    from the checkpoint's ``ctc.ctc_lo``; train with ``data.tokenizer: subword``). The default
    ``head: char`` is a new 28-character layer."""

    def __init__(self, model_cfg: dict | None = None) -> None:
        super().__init__()
        cfg = model_cfg or {}
        self.visual_encoder = _avsr_from_cfg(cfg)
        head = cfg.get("head", "char")
        if head == "avsr_ctc":
            from lipscribe.data import subword
            self.ctc_head = LinearCTC(self.visual_encoder.out_dim, float(cfg.get("dropout", 0.1)),
                                      vocab_size=subword.VOCAB_SIZE)
            v = cfg.get("visual_encoder_cfg") or {}
            path = _resolve(v.get("checkpoint", DEFAULT_AVSR_CKPT)) if v.get("pretrained", True) else None
            if path is not None:             # (a resumed or fused model gets its weights elsewhere)
                sd = torch.load(path, map_location="cpu", weights_only=True)
                self.ctc_head.proj.load_state_dict({"weight": sd["ctc.ctc_lo.weight"], "bias": sd["ctc.ctc_lo.bias"]})
                print(f"Loaded Auto-AVSR word-piece CTC layer from {path}")
        elif head == "char":
            self.ctc_head = LinearCTC(self.visual_encoder.out_dim, float(cfg.get("dropout", 0.1)))
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, batch: dict) -> dict:
        T = batch["video"].size(2)
        lengths = batch["lengths"].to(batch["video"].device).clamp(min=1, max=T)
        h = self.visual_encoder(batch["video"], lengths)
        return {"log_probs": self.ctc_head(h), "lengths": lengths}


class V4Fusion(nn.Module):
    """WavLM + Auto-AVSR encoders joined with Model P's cross-modal attention and gate."""

    def __init__(self, model_cfg: dict | None = None) -> None:
        super().__init__()
        from lipscribe.models import _extract_modality_dropout
        from lipscribe.models.fusion_attention import AttentionFusion

        cfg = model_cfg or {}
        p_audio, p_visual = _extract_modality_dropout(cfg, 0.25, 0.15)
        embed = int(cfg.get("embed_dim", 256))
        self.audio_encoder = _wavlm_from_cfg(cfg)
        self.visual_encoder = _avsr_from_cfg(cfg)
        self.fusion = AttentionFusion(in_dim=768, embed_dim=embed, num_heads=int(cfg.get("num_heads", 4)),
                                      attn_dropout=float(cfg.get("attn_dropout", 0.1)),
                                      p_audio_drop=p_audio, p_visual_drop=p_visual)
        self.ctc_head = CTCHead(input_dim=embed, dropout=float(cfg.get("dropout", 0.1)))

    def forward(self, batch: dict) -> dict:
        a, a_lens = self.audio_encoder(batch["audio"], batch["audio_lengths"])     # 50 fps
        T = batch["video"].size(2)
        v_lens = batch["lengths"].to(a.device).clamp(min=1, max=T)
        v = self.visual_encoder(batch["video"], v_lens).repeat_interleave(2, dim=1)  # 25 -> 50 fps
        t = min(a.size(1), v.size(1))
        a, v = a[:, :t], v[:, :t]
        lengths = torch.minimum(a_lens, v_lens * 2).clamp(min=1, max=t)
        mask = frames_mask(lengths, t)
        video_valid = batch.get("video_valid")
        if video_valid is not None:
            video_valid = video_valid.to(a.device, torch.bool).repeat_interleave(2, dim=1)[:, :t]
            if video_valid.size(1) < t:
                video_valid = nn.functional.pad(video_valid, (0, t - video_valid.size(1)))
        fused, gate, w = self.fusion(a, v, mask=mask, video_valid=video_valid, lengths=lengths,
                                     drop_modality=batch.get("drop_modality"), return_weights=True)
        return {"log_probs": self.ctc_head(fused, lengths), "lengths": lengths, "gate": gate,
                "attn_a2v": w["a2v"], "attn_v2a": w["v2a"]}


class V4FusionWP(nn.Module):
    """Flix v4 fusion v2: both streams on Auto-AVSR's word pieces at 25 frames/s.

    v4_fusion ignored the lips: swapping in another clip's video did not change its output.
    The causes were a new character head (the visual encoder had been trained for word
    pieces), the pretrained audio taking over in the first epoch, and a scalar gate whose
    "visual" branch already carried attended audio. This model fixes those:

    * **One vocabulary.** Every head is a 5,049-way word-piece CTC layer at 25 frames/s, the
      same space as the visual encoder and its pretrained layer (``ctc.ctc_lo``).
    * **Audio at 25 frames/s.** Each pair of 20 ms WavLM frames is concatenated and projected
      (1536 -> 768), so one audio frame lines up with one video frame.
    * **Lip-anchored fusion.** ``f = v + MLP([a; v])``. The MLP's last layer starts at zero
      and the fused head starts as a copy of the visual head, so at step 0 the model *is*
      visual_c (lips only). Training can only add audio on top of a working lip-reader; the
      lips are not left behind while audio wins.
    * Then ``fusion_layers`` Transformer layers (pre-norm) over the fused sequence, and the
      fused CTC head.
    * **Auxiliary CTC losses** (``aux_ctc: {audio: w, visual: w}``) on each stream's own head,
      computed on the encoder outputs before modality dropout. They keep both streams
      discriminative. ``train.py`` adds ``sum(w * CTC(aux))`` to the fused loss.
    * **Modality dropout.** A whole stream is replaced by a learned 768-d vector (``miss_a`` /
      ``miss_v``) per utterance. Test-time ``drop_modality`` does the same, and so does a clip
      whose face is missing in > 50 % of frames. With ``modality_dropout.visual: 1.0`` this is
      the audio-only twin: identical parameters, data and noise, with the lips never seen.

    ``init_heads_from``: a visual_c checkpoint whose ``ctc_head`` initialises both the visual
    head and the fused head. Encoders come from ``init_from`` as usual (train.py).
    """

    def __init__(self, model_cfg: dict | None = None) -> None:
        super().__init__()
        from lipscribe.data import subword

        cfg = model_cfg or {}
        md = cfg.get("modality_dropout") or {}
        self.p_audio_drop = float(md.get("audio", 0.3))
        self.p_visual_drop = float(md.get("visual", 0.1))
        if self.p_audio_drop + self.p_visual_drop > 1.0 + 1e-9:
            raise ValueError("modality_dropout audio + visual must be <= 1")
        aux = cfg.get("aux_ctc") or {}
        self.aux_w = {"audio": float(aux.get("audio", 0.3)), "visual": float(aux.get("visual", 0.3))}
        d, drop, V = 768, float(cfg.get("dropout", 0.1)), subword.VOCAB_SIZE

        self.audio_encoder = _wavlm_from_cfg(cfg)
        self.visual_encoder = _avsr_from_cfg(cfg)
        self.audio_pool = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.miss_a = nn.Parameter(torch.randn(d) * 0.02)
        self.miss_v = nn.Parameter(torch.randn(d) * 0.02)
        self.fuse_mlp = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                      nn.Dropout(drop), nn.Linear(d, d))
        nn.init.zeros_(self.fuse_mlp[-1].weight); nn.init.zeros_(self.fuse_mlp[-1].bias)
        n_layers = int(cfg.get("fusion_layers", 2))
        layer = nn.TransformerEncoderLayer(d, nhead=int(cfg.get("num_heads", 8)), dim_feedforward=4 * d,
                                           dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.fusion_encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False) if n_layers else None
        if self.fusion_encoder is not None:   # pre-norm residual layers start as the identity
            for lyr in self.fusion_encoder.layers:
                for lin in (lyr.self_attn.out_proj, lyr.linear2):
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.fused_head = LinearCTC(d, drop, vocab_size=V)
        self.audio_head = LinearCTC(d, drop, vocab_size=V)
        self.visual_head = LinearCTC(d, drop, vocab_size=V)
        src = _resolve(cfg.get("init_heads_from"))
        if src is not None:
            sd = torch.load(src, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            head = {"weight": sd["ctc_head.proj.weight"], "bias": sd["ctc_head.proj.bias"]}
            self.visual_head.proj.load_state_dict(head)
            self.fused_head.proj.load_state_dict(head)
            print(f"Initialised visual_head and fused_head from {src} (ctc_head)")
        full = _resolve(cfg.get("init_model_from"))     # fine-tune a whole earlier v4_fusion_wp run
        if full is not None:
            sd = torch.load(full, map_location="cpu", weights_only=False)
            self.load_state_dict(sd.get("model", sd), strict=True)
            print(f"Initialised the whole model from {full}")

    def _pool_audio(self, a: torch.Tensor, a_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.size(1) % 2:
            a = nn.functional.pad(a, (0, 0, 0, 1))
        B, T, D = a.shape
        return self.audio_pool(a.reshape(B, T // 2, 2 * D)), (a_lens + 1) // 2

    def _drop_masks(self, B: int, device, drop_modality, video_valid, lengths, t):
        drop_a = torch.zeros(B, dtype=torch.bool, device=device)
        drop_v = torch.zeros(B, dtype=torch.bool, device=device)
        if drop_modality == "audio":
            drop_a[:] = True
        elif drop_modality == "visual":
            drop_v[:] = True
        elif self.p_visual_drop >= 1.0:       # audio-only twin: never sees the lips, also at test
            drop_v[:] = True
        elif self.training:
            u = torch.rand(B, device=device)
            drop_a = u < self.p_audio_drop
            drop_v = (u >= self.p_audio_drop) & (u < self.p_audio_drop + self.p_visual_drop)
        if video_valid is not None and drop_modality != "audio":
            vv = video_valid.to(device, torch.bool)[:, :t]
            if vv.size(1) < t:
                vv = nn.functional.pad(vv, (0, t - vv.size(1)), value=True)
            m = frames_mask(lengths, t)
            no_face = (vv & m).sum(1).float() < 0.5 * m.sum(1).clamp(min=1).float()
            drop_v = drop_v | (no_face & ~drop_a)
        return drop_a, drop_v

    def forward(self, batch: dict) -> dict:
        a50, a50_lens = self.audio_encoder(batch["audio"], batch["audio_lengths"])
        a, a_lens = self._pool_audio(a50, a50_lens)                                  # 25 fps
        T = batch["video"].size(2)
        v_lens = batch["lengths"].to(a.device).clamp(min=1, max=T)
        v = self.visual_encoder(batch["video"], v_lens)                              # 25 fps
        t = min(a.size(1), v.size(1))
        a, v = a[:, :t], v[:, :t]
        lengths = torch.minimum(a_lens, v_lens).clamp(min=1, max=t)
        mask = frames_mask(lengths, t)
        B = a.size(0)
        drop_a, drop_v = self._drop_masks(B, a.device, batch.get("drop_modality"),
                                          batch.get("video_valid"), lengths, t)
        af = torch.where(drop_a.view(B, 1, 1), self.miss_a.to(a.dtype).expand_as(a), a)
        vf = torch.where(drop_v.view(B, 1, 1), self.miss_v.to(v.dtype).expand_as(v), v)
        f = vf + self.fuse_mlp(torch.cat([af, vf], dim=-1))
        if self.fusion_encoder is not None:
            f = self.fusion_encoder(f, src_key_padding_mask=~mask)
        out = {"log_probs": self.fused_head(f), "lengths": lengths}
        if self.training or batch.get("return_aux"):
            out["aux_ctc"] = [(self.aux_w["audio"], self.audio_head(a)),
                              (self.aux_w["visual"], self.visual_head(v))]
        return out


class V4FusionGated(V4FusionWP):
    """Flix v4 fusion v3: symmetric, reliability-gated fusion of two finished single-stream models.

    v4_fusion ignored the lips (fusion was audio-anchored in practice). v4_fusion_wp was
    lip-anchored (``f = v + MLP``) and over-trusted the lips, so it was worse on clean audio than its
    audio-only twin. Fine-tuning could not undo the built-in residual. Here neither stream is
    built in:

    * **Streams.** The audio side (WavLM + 25 fps pooling + word-piece head) comes from the
      audio-only twin, and the lip side (Auto-AVSR encoder + word-piece head) from fusion_wp.
      ``init_streams_from: {audio: RUN/best.pt, visual: RUN/best.pt}``. Usually frozen
      (``model.freeze``), so only the fusion is learned and neither stream can degrade.
    * **Reliability features** per frame from each stream's own (frozen) CTC head: normalised
      entropy and blank probability, 4 numbers, computed without gradient. A dropped or missing
      stream reports entropy 1 and blank 0.
    * **Per-channel gate.** ``a' = LN(W_a a)`` and ``v' = LN(W_v v)``; then
      ``g = sigmoid(W_g [a'; v'; r])``, 768 channels per frame with bias 0 (an even split at
      the start); ``f = g * a' + (1 - g) * v' + MLP([a'; v'])``. The MLP lets the streams
      combine beyond a weighted average and starts at zero.
    * ``fusion_layers`` pre-norm Transformer layers, then a new word-piece CTC head.
    * Output ``gate`` ``[B, T]`` is the mean of g over channels (1 = audio).
    """

    def __init__(self, model_cfg: dict | None = None) -> None:
        nn.Module.__init__(self)
        from lipscribe.data import subword

        cfg = model_cfg or {}
        md = cfg.get("modality_dropout") or {}
        self.p_audio_drop = float(md.get("audio", 0.15))
        self.p_visual_drop = float(md.get("visual", 0.15))
        if self.p_audio_drop + self.p_visual_drop > 1.0 + 1e-9:
            raise ValueError("modality_dropout audio + visual must be <= 1")
        aux = cfg.get("aux_ctc") or {}
        self.aux_w = {"audio": float(aux.get("audio", 0.0)), "visual": float(aux.get("visual", 0.0))}
        d, drop, V = 768, float(cfg.get("dropout", 0.1)), subword.VOCAB_SIZE
        self.log_v = float(torch.log(torch.tensor(float(V))))

        self.audio_encoder = _wavlm_from_cfg(cfg)
        self.visual_encoder = _avsr_from_cfg(cfg)
        self.audio_pool = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.audio_head = LinearCTC(d, drop, vocab_size=V)
        self.visual_head = LinearCTC(d, drop, vocab_size=V)
        self.miss_a = nn.Parameter(torch.randn(d) * 0.02)
        self.miss_v = nn.Parameter(torch.randn(d) * 0.02)

        self.proj_a = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.proj_v = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.gate = nn.Linear(2 * d + 4, d)
        nn.init.zeros_(self.gate.bias)
        self.synergy = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        nn.init.zeros_(self.synergy[-1].weight); nn.init.zeros_(self.synergy[-1].bias)
        n_layers = int(cfg.get("fusion_layers", 3))
        layer = nn.TransformerEncoderLayer(d, nhead=int(cfg.get("num_heads", 8)), dim_feedforward=4 * d,
                                           dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.fusion_encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False) if n_layers else None
        self.fused_head = LinearCTC(d, drop, vocab_size=V)

        src = cfg.get("init_streams_from") or {}
        parts = {"audio": ("audio_encoder.", "audio_pool.", "audio_head."),
                 "visual": ("visual_encoder.", "visual_head.")}
        own = self.state_dict()
        for side, prefixes in parts.items():
            path = _resolve(src.get(side))
            if path is None:
                continue
            sd = torch.load(path, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            take = {k: v for k, v in sd.items() if k.startswith(prefixes)}
            need = [k for k in own if k.startswith(prefixes)]
            missing = [k for k in need if k not in take]
            if missing:
                raise ValueError(f"init_streams_from.{side}: {path} lacks {missing[:3]}")
            self.load_state_dict(take, strict=False)
            print(f"Initialised {side} stream ({', '.join(p[:-1] for p in prefixes)}) from {path} ({len(take)} tensors)")

    def _reliability(self, head: LinearCTC, x: torch.Tensor, dropped: torch.Tensor):
        with torch.no_grad():
            lp = head.proj(x).float().log_softmax(-1)                 # [B, T, V], no dropout
            ent = -(lp.exp() * lp).sum(-1) / self.log_v
            blank = lp[..., 0].exp()
            ent = torch.where(dropped.view(-1, 1), torch.ones_like(ent), ent)
            blank = torch.where(dropped.view(-1, 1), torch.zeros_like(blank), blank)
        return lp.transpose(0, 1), torch.stack([ent, blank], dim=-1)

    def forward(self, batch: dict) -> dict:
        a50, a50_lens = self.audio_encoder(batch["audio"], batch["audio_lengths"])
        a, a_lens = self._pool_audio(a50, a50_lens)                                  # 25 fps
        T = batch["video"].size(2)
        v_lens = batch["lengths"].to(a.device).clamp(min=1, max=T)
        v = self.visual_encoder(batch["video"], v_lens)                              # 25 fps
        t = min(a.size(1), v.size(1))
        a, v = a[:, :t], v[:, :t]
        lengths = torch.minimum(a_lens, v_lens).clamp(min=1, max=t)
        mask = frames_mask(lengths, t)
        B = a.size(0)
        drop_a, drop_v = self._drop_masks(B, a.device, batch.get("drop_modality"),
                                          batch.get("video_valid"), lengths, t)
        lp_a, r_a = self._reliability(self.audio_head, a, drop_a)
        lp_v, r_v = self._reliability(self.visual_head, v, drop_v)
        af = torch.where(drop_a.view(B, 1, 1), self.miss_a.to(a.dtype).expand_as(a), a)
        vf = torch.where(drop_v.view(B, 1, 1), self.miss_v.to(v.dtype).expand_as(v), v)
        ap, vp = self.proj_a(af), self.proj_v(vf)
        g = torch.sigmoid(self.gate(torch.cat([ap, vp, r_a.to(ap.dtype), r_v.to(ap.dtype)], dim=-1)))
        f = g * ap + (1 - g) * vp + self.synergy(torch.cat([ap, vp], dim=-1))
        if self.fusion_encoder is not None:
            f = self.fusion_encoder(f, src_key_padding_mask=~mask)
        out = {"log_probs": self.fused_head(f), "lengths": lengths, "gate": g.mean(-1)}
        if self.training and (self.aux_w["audio"] > 0 or self.aux_w["visual"] > 0):
            out["aux_ctc"] = [(self.aux_w["audio"], self.audio_head(a)),
                              (self.aux_w["visual"], self.visual_head(v))]
        return out
