from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from emg2qwerty.charset import charset
from emg2qwerty.models.common import CTCModelBase
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, C)
        if x.size(0) > self.pe.size(0):
            self.pe = self._build_pe(x.size(0)).to(x.device)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerCTCModule(CTCModelBase, pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_ff_dim: int,
        transformer_dropout: float,
        temporal_stride: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        frontend_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * electrode_channels),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.input_proj = nn.Linear(frontend_features, transformer_dim)
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        self.temporal_stride = temporal_stride
        self.temporal_downsample = (
            nn.Conv1d(
                in_channels=transformer_dim,
                out_channels=transformer_dim,
                kernel_size=3,
                stride=temporal_stride,
                padding=1,
            )
            if temporal_stride > 1
            else nn.Identity()
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=transformer_dim,
            dropout=transformer_dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self._init_ctc_common(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.input_proj(x)
        if self.temporal_stride > 1:
            # (T, N, C) -> (N, C, T) for Conv1d over temporal axis.
            x = x.permute(1, 2, 0)
            x = self.temporal_downsample(x)
            x = x.permute(2, 0, 1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.classifier(x)
