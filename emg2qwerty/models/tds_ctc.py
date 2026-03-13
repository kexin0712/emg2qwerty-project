from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from emg2qwerty.charset import charset
from emg2qwerty.models.common import CTCModelBase
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)


class TDSConvCTCModule(CTCModelBase, pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * electrode_channels),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self._init_ctc_common(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
