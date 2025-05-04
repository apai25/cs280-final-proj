import torch
import torch.nn.functional as F
from torch import nn

from configs.model_config import ModelConfig


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(UNet, self).__init__()
        self.cfg = cfg

        all_channel_inputs = []
        obs_cond_channels = cfg.input_channels * cfg.horizon

        # Define unet arch
        self.unet_enc = nn.ModuleList()
        in_channels = cfg.input_channels
        for i, out_channels in enumerate(cfg.hidden_channels):
            if i in cfg.obs_cond_stages:
                in_channels += obs_cond_channels
            all_channel_inputs.append(in_channels)
            self.unet_enc.append(self._down_conv_block(in_channels, out_channels))
            in_channels = out_channels

        if len(self.unet_enc) in cfg.obs_cond_stages:
            in_channels += obs_cond_channels
        all_channel_inputs.append(in_channels)
        self.bottleneck = self._conv_block(in_channels, cfg.bottleneck_channels)
        in_channels = cfg.bottleneck_channels

        self.unet_dec = nn.ModuleList()
        for i, out_channels in enumerate(reversed(cfg.hidden_channels)):
            i_rel = len(cfg.hidden_channels) + 1 + i
            if i_rel in cfg.obs_cond_stages:
                in_channels += obs_cond_channels
            in_channels += out_channels  # residual conn
            all_channel_inputs.append(in_channels)
            self.unet_dec.append(self._up_conv_block(in_channels, out_channels))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(
            in_channels, cfg.output_channels, kernel_size=1
        )  # no conditioning for final conv

        # Time MLPs
        self.t_mlps = nn.ModuleList()
        for s in cfg.t_cond_stages:
            self.t_mlps.append(self._mlp(self.cfg.time_embed_dim, all_channel_inputs[s]))

        # Action MLPs
        self.act_mlps = nn.ModuleList()
        for s in cfg.act_cond_stages:
            self.act_mlps.append(
                self._mlp(cfg.action_dim * cfg.horizon, all_channel_inputs[s])
            )

        # Obs Encs
        self.obs_enc = nn.ModuleList()
        for _ in cfg.obs_cond_stages:
            self.obs_enc.append(
                self._conv_block(
                    cfg.input_channels * cfg.horizon, cfg.input_channels * cfg.horizon
                )
            )

        # Helpers for forward
        self.t_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.t_cond_stages)}
        self.act_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.act_cond_stages)}
        self.obs_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.obs_cond_stages)}

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
    
    def _sin_embed(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.cfg.time_embed_dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def _conditioner(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
        global_idx: int,
        drop_cond: torch.Tensor,
    ) -> torch.Tensor:
        if drop_cond is not None and drop_cond.any():
            context_acts = context_acts.clone()
            context_obs = context_obs.clone()
            context_acts[drop_cond] = 0
            context_obs[drop_cond] = 0

        if global_idx in self.cfg.obs_cond_stages:
            cond_idx = self.obs_cond_stage_to_idx[global_idx]
            cond_enc = self.obs_enc[cond_idx]
            context_obs = cond_enc(context_obs)

            if context_obs.shape[-2:] != x.shape[-2:]:
                context_obs = F.interpolate(
                    context_obs, size=x.shape[-2:], mode="bilinear"
                )
            x = torch.cat([x, context_obs], dim=1)

        if global_idx in self.cfg.t_cond_stages:
            cond_idx = self.t_cond_stage_to_idx[global_idx]
            t_mlp = self.t_mlps[cond_idx]
            t = t_mlp(self._sin_embed(t))
            t = t.view(t.shape[0], t.shape[1], 1, 1)
            x = x + t

        if global_idx in self.cfg.act_cond_stages:
            cond_idx = self.act_cond_stage_to_idx[global_idx]
            act_mlp = self.act_mlps[cond_idx]
            context_acts = act_mlp(context_acts)
            context_acts = context_acts.view(
                context_acts.shape[0], context_acts.shape[1], 1, 1
            )
            x = x + context_acts
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
        drop_cond: torch.Tensor,
    ) -> torch.Tensor:
        enc_outs = []
        eps_pred = x_t
        for i, layer in enumerate(self.unet_enc):
            eps_pred = self._conditioner(
                eps_pred, t, context_acts, context_obs, i, drop_cond
            )
            eps_pred = layer(eps_pred)
            enc_outs.append(eps_pred)

        eps_pred = self._conditioner(
            eps_pred, t, context_acts, context_obs, len(self.unet_enc), drop_cond
        )
        eps_pred = self.bottleneck(eps_pred)

        for i, layer in enumerate(self.unet_dec):
            rel_i = i + len(self.unet_enc) + 1

            skip = enc_outs[-(i + 1)]

            eps_pred = torch.cat([eps_pred, skip], dim=1)
            eps_pred = self._conditioner(
                eps_pred, t, context_acts, context_obs, rel_i, drop_cond
            )
            eps_pred = layer(eps_pred)

        eps_pred = self.final_conv(eps_pred)

        return eps_pred
