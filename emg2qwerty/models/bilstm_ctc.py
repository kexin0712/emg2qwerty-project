from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from emg2qwerty.charset import charset
from emg2qwerty.models.common import CTCModelBase
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class BiLSTMCTCModule(CTCModelBase, pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        output_dropout: float,
        use_layer_norm: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        lstm_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * electrode_channels),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.encoder_norm = (
            nn.LayerNorm(2 * lstm_hidden_size) if use_layer_norm else nn.Identity()
        )
        self.output_dropout = nn.Dropout(output_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self._init_ctc_common(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x, _ = self.encoder(x)
        x = self.encoder_norm(x)
        x = self.output_dropout(x)
        return self.classifier(x)
