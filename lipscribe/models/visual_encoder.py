"""Visual encoders for LipScribe (implementation plan, Section 5 / Table 6).

Two front ends share one interface, ``[B, 1, T, 96, 96] -> [B, T, 512]``:

* ``VisualEncoder`` (``conv3d_bilstm``): three Conv3d blocks -> Linear 512 -> 2-layer BiLSTM.
  This is the original V1 encoder. Its parameter names are unchanged, so the
  first B2/B3 checkpoints still load when the new options are left at their
  defaults (``spatial_dropout=0``, ``input_norm="none"``).
* ``ResNetVisualEncoder`` (``resnet18_bilstm``): one Conv3d (5x7x7) followed by a
  per-frame 2D ResNet-18 trunk -> 512 -> 2-layer BiLSTM. This is the standard
  lipreading front end (Stafylakis & Tzimiropoulos 2017; Martinez et al. 2020)
  and generalises much better to unseen speakers than three plain conv blocks.

``input_norm="clip"`` standardises each clip to zero mean and unit variance over
its real (unpadded) frames, which removes per-speaker brightness and contrast
differences before the first convolution.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _check_video(video: torch.Tensor, name: str) -> None:
    if video.dim() != 5 or video.size(1) != 1 or video.size(3) != 96 or video.size(4) != 96:
        raise ValueError(f"{name} expects [B, 1, T, 96, 96], got shape {tuple(video.shape)}")


def clip_normalise(video: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Zero-mean, unit-variance per clip over real frames; padded frames stay 0."""
    B, _, T, _, _ = video.shape
    if lengths is None:
        mask = torch.ones(B, T, dtype=torch.bool, device=video.device)
    else:
        mask = torch.arange(T, device=video.device).unsqueeze(0) < lengths.to(video.device).unsqueeze(1)
    m = mask.view(B, 1, T, 1, 1).to(video.dtype)
    n = m.sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1) * video.size(3) * video.size(4)
    mean = (video * m).sum(dim=(1, 2, 3, 4), keepdim=True) / n
    var = (((video - mean) * m) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True) / n
    return (video - mean) / var.clamp(min=1e-6).sqrt() * m


class _TemporalBiLSTM(nn.Module):
    """2-layer BiLSTM (256 per direction) that skips padded frames."""

    def __init__(self, input_size: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        T = features.size(1)
        if lengths is None:
            return self.lstm(features)[0]
        lengths = lengths.to(device=features.device, dtype=torch.long).clamp(min=1, max=T)
        packed = pack_padded_sequence(features, lengths.detach().cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        return pad_packed_sequence(out, batch_first=True, total_length=T)[0]


class VisualEncoder(nn.Module):
    """Three-block 3D-CNN followed by a projection and a 2-layer BiLSTM."""

    def __init__(self, dropout: float = 0.3, spatial_dropout: float = 0.0,
                 input_norm: str = "none") -> None:
        super().__init__()
        if input_norm not in ("none", "clip"):
            raise ValueError(f"input_norm must be 'none' or 'clip', got {input_norm!r}")
        self.input_norm = input_norm
        layers: list[nn.Module] = []
        specs = [(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
                 (32, 64, (3, 5, 5), 1, (1, 2, 2)),
                 (64, 96, (3, 3, 3), 1, 1)]
        for c_in, c_out, k, s, p in specs:
            layers += [nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                       nn.BatchNorm3d(c_out), nn.ReLU(inplace=True),
                       nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
            if spatial_dropout > 0:
                # LipNet-style spatial dropout: drops whole feature maps.
                layers.append(nn.Dropout3d(spatial_dropout))
        self.conv = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(96 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        video: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[B, 1, T, 96, 96]`` video into ``[B, T, 512]`` features."""
        _check_video(video, "VisualEncoder")
        if self.input_norm == "clip":
            video = clip_normalise(video, lengths)

        batch_size, _, time_steps, _, _ = video.shape
        features = self.conv(video)
        features = features.permute(0, 2, 1, 3, 4).reshape(batch_size, time_steps, -1)
        features = self.projection(features)

        if lengths is None:
            encoded, _ = self.lstm(features)
            return encoded

        lengths = lengths.to(device=video.device, dtype=torch.long).clamp(min=1, max=time_steps)
        packed = pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=time_steps
        )
        return encoded


class _BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                      nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetVisualEncoder(nn.Module):
    """Conv3d (5x7x7) front end + per-frame ResNet-18 trunk + 2-layer BiLSTM."""

    def __init__(self, dropout: float = 0.3, spatial_dropout: float = 0.0,
                 input_norm: str = "clip") -> None:
        super().__init__()
        if input_norm not in ("none", "clip"):
            raise ValueError(f"input_norm must be 'none' or 'clip', got {input_norm!r}")
        self.input_norm = input_norm
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )                                                     # 96 -> 48 -> 24
        blocks: list[nn.Module] = []
        c_in = 64
        for c_out, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):  # 24 -> 24, 12, 6, 3
            blocks += [_BasicBlock(c_in, c_out, stride), _BasicBlock(c_out, c_out)]
            c_in = c_out
        self.trunk = nn.Sequential(*blocks)
        self.spatial_dropout = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.temporal = _TemporalBiLSTM(512)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():
            if isinstance(m, _BasicBlock):
                nn.init.zeros_(m.bn2.weight)  # residual branches start as identity

    def forward(self, video: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _check_video(video, "ResNetVisualEncoder")
        if self.input_norm == "clip":
            video = clip_normalise(video, lengths)
        B, _, T, _, _ = video.shape
        x = self.frontend(video)                              # [B, 64, T, 24, 24]
        x = x.transpose(1, 2).reshape(B * T, 64, x.size(3), x.size(4))
        x = self.spatial_dropout(self.trunk(x))
        x = self.pool(x).flatten(1).view(B, T, 512)
        return self.temporal(self.dropout(x), lengths)


def build_visual_encoder(model_cfg: dict | None = None, dropout: float = 0.3) -> nn.Module:
    """Visual encoder named by ``model.visual_encoder`` (default ``conv3d_bilstm``)."""
    cfg = model_cfg or {}
    name = cfg.get("visual_encoder", "conv3d_bilstm")
    spatial = float(cfg.get("spatial_dropout", 0.0))
    if name == "conv3d_bilstm":
        return VisualEncoder(dropout=dropout, spatial_dropout=spatial,
                             input_norm=cfg.get("input_norm", "none"))
    if name == "resnet18_bilstm":
        return ResNetVisualEncoder(dropout=dropout, spatial_dropout=spatial,
                                   input_norm=cfg.get("input_norm", "clip"))
    raise ValueError(f"Unknown visual_encoder {name!r} (conv3d_bilstm or resnet18_bilstm)")
