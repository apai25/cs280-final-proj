from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from configs.model_config import ModelConfig


@dataclass
class ModelOutput:
    pred: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(Model, self).__init__()
        self.cfg = cfg

        all_channel_inputs = []

        # Define unet arch
        self.unet_enc = nn.ModuleList()
        in_channels = cfg.input_channels
        for out_channels in cfg.hidden_channels:
            all_channel_inputs.append(in_channels)
            self.unet_enc.append(self._down_conv_block(in_channels, out_channels))
            in_channels = out_channels

        all_channel_inputs.append(in_channels)
        self.bottleneck = self._conv_block(in_channels, cfg.bottleneck_channels)
        in_channels = cfg.bottleneck_channels

        self.unet_dec = nn.ModuleList()
        for out_channels in reversed(cfg.hidden_channels):
            all_channel_inputs.append(in_channels + out_channels)
            self.unet_dec.append(
                self._up_conv_block(in_channels + out_channels, out_channels)
            )
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, cfg.output_channels, kernel_size=1)

        # Conditioning MLPs
        self.act_mlps = nn.ModuleList()
        for s in cfg.cond_stages:
            self.act_mlps.append(
                self._mlp(
                    cfg.action_dim * cfg.input_channels // 3, all_channel_inputs[s]
                )
            )

        self.t_mlps = nn.ModuleList()
        for s in cfg.cond_stages:
            self.t_mlps.append(self._mlp(1, all_channel_inputs[s]))

        self.cond_stage_to_idx = {s: i for i, s in enumerate(cfg.cond_stages)}

        # Loss
        if self.cfg.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {self.cfg.loss_fn}")

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if self.cfg.batch_norm else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if self.cfg.batch_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _up_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=self.cfg.pooling_kernel_size,
                stride=self.cfg.pooling_stride,
            ),
        ] + list(self._conv_block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def _down_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        layers = list(self._conv_block(in_channels, out_channels)) + [
            nn.MaxPool2d(
                kernel_size=self.cfg.pooling_kernel_size,
                stride=self.cfg.pooling_stride,
            )
        ]
        return nn.Sequential(*layers)

    def _mlp(self, in_dim: int, out_dim: int) -> nn.Module:
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        ]
        return nn.Sequential(*layers)

    def forward(
        self,
        imgs: torch.Tensor,
        acts: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> ModelOutput:
        enc_outs = []
        for i, layer in enumerate(self.unet_enc):
            if i in self.cfg.cond_stages:
                idx = self.cond_stage_to_idx[i]
                act_out = self.act_mlps[idx](acts)
                t_out = self.t_mlps[idx](t)
                act_out = act_out.view(act_out.size(0), act_out.size(1), 1, 1)
                t_out = t_out.view(t_out.size(0), t_out.size(1), 1, 1)
                imgs = imgs + act_out + t_out
            imgs = layer(imgs)
            enc_outs.append(imgs)

        imgs = self.bottleneck(imgs)

        for i, layer in enumerate(self.unet_dec):
            imgs = torch.cat([imgs, enc_outs[-(i + 1)]], dim=1)
            rel_i = i + len(self.unet_enc) + 1
            if rel_i in self.cfg.cond_stages:
                idx = self.cond_stage_to_idx[rel_i]
                act_out = self.act_mlps[idx](acts)
                t_out = self.t_mlps[idx](t)
                act_out = act_out.view(act_out.size(0), act_out.size(1), 1, 1)
                t_out = t_out.view(t_out.size(0), t_out.size(1), 1, 1)
                imgs = imgs + act_out + t_out
            imgs = layer(imgs)

        pred = self.final_conv(imgs)

        loss = None
        if y is not None:
            loss = self.loss_fn(pred, y)

        return ModelOutput(pred=pred, loss=loss)
