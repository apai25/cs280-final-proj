import torch
import torch.nn.functional as F
from torch import nn

from configs.model_config import ModelConfig
from src.models.blocks import MLP, ConvBlock, DownConvBlock, UpConvBlock


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(UNet, self).__init__()
        self.cfg = cfg

        all_channel_inputs = []
        obs_cond_channels = cfg.img_channels * cfg.horizon

        # Define unet arch
        self.unet_enc = nn.ModuleList()
        in_channels = cfg.img_channels
        for i, out_channels in enumerate(cfg.hidden_channels):
            if i in cfg.obs_cond_stages:
                in_channels += obs_cond_channels
            all_channel_inputs.append(in_channels)
            self.unet_enc.append(
                DownConvBlock(in_channels, out_channels, cfg.pooling_kernel_size, cfg.pooling_stride, cfg.dropout, cfg.batch_norm)
            )
            in_channels = out_channels

        if len(self.unet_enc) in cfg.obs_cond_stages:
            in_channels += obs_cond_channels
        all_channel_inputs.append(in_channels)
        self.bottleneck = ConvBlock(
            in_channels, cfg.bottleneck_channels, cfg.dropout, cfg.batch_norm
        )
        in_channels = cfg.bottleneck_channels

        self.unet_dec = nn.ModuleList()
        for i, out_channels in enumerate(reversed(cfg.hidden_channels)):
            i_rel = len(cfg.hidden_channels) + 1 + i
            if i_rel in cfg.obs_cond_stages:
                in_channels += obs_cond_channels
            in_channels += out_channels  # residual conn
            all_channel_inputs.append(in_channels)
            self.unet_dec.append(
                UpConvBlock(in_channels, out_channels, cfg.pooling_kernel_size, cfg.pooling_stride, cfg.dropout, cfg.batch_norm)
            )
            in_channels = out_channels

        self.final_conv = nn.Conv2d(
            in_channels, cfg.img_channels, kernel_size=1
        )  # no conditioning for final conv

        # Conditioning Layers
        self.t_mlps = nn.ModuleList(
            MLP(1, all_channel_inputs[s]) for s in cfg.t_cond_stages
        )
        self.act_mlps = nn.ModuleList(
            MLP(cfg.action_dim * cfg.horizon, all_channel_inputs[s])
            for s in cfg.act_cond_stages
        )
        self.obs_encs = nn.ModuleList(
            ConvBlock(obs_cond_channels, obs_cond_channels, cfg.dropout, cfg.batch_norm)
            for _ in cfg.obs_cond_stages
        )

        # Helpers for forward
        self.t_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.t_cond_stages)}
        self.act_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.act_cond_stages)}
        self.obs_cond_stage_to_idx = {s: i for i, s in enumerate(cfg.obs_cond_stages)}

    def _conditioner(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
        global_idx: int,
        drop_cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        if drop_cond_mask.any():
            context_acts = context_acts.clone()
            context_obs = context_obs.clone()
            context_acts[drop_cond_mask] = 0
            context_obs[drop_cond_mask] = 0

        if global_idx in self.cfg.obs_cond_stages:
            cond_idx = self.obs_cond_stage_to_idx[global_idx]
            cond_enc = self.obs_encs[cond_idx]
            context_obs = cond_enc(context_obs)

            if context_obs.shape[-2:] != x.shape[-2:]:
                context_obs = F.interpolate(
                    context_obs, size=x.shape[-2:], mode="bilinear"
                )
            x = torch.cat([x, context_obs], dim=1)

        if global_idx in self.cfg.t_cond_stages:
            cond_idx = self.t_cond_stage_to_idx[global_idx]
            t_mlp = self.t_mlps[cond_idx]
            t = t_mlp(t)
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
        drop_cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc_outs = []
        eps_pred = x_t
        for i, layer in enumerate(self.unet_enc):
            eps_pred = self._conditioner(
                eps_pred, t, context_acts, context_obs, i, drop_cond_mask
            )
            eps_pred = layer(eps_pred)
            enc_outs.append(eps_pred)

        eps_pred = self._conditioner(
            eps_pred, t, context_acts, context_obs, len(self.unet_enc), drop_cond_mask
        )
        eps_pred = self.bottleneck(eps_pred)

        for i, layer in enumerate(self.unet_dec):
            rel_i = i + len(self.unet_enc) + 1

            skip = enc_outs[-(i + 1)]

            eps_pred = torch.cat([eps_pred, skip], dim=1)
            eps_pred = self._conditioner(
                eps_pred, t, context_acts, context_obs, rel_i, drop_cond_mask
            )
            eps_pred = layer(eps_pred)

        eps_pred = self.final_conv(eps_pred)

        return eps_pred
