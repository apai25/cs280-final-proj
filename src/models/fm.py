from typing import Optional

import torch
from torch import nn

from src.models.unet import UNet


class FM(nn.Module):
    def __init__(self, cfg):
        super(FM, self).__init__()
        self.cfg = cfg
        self.unet = UNet(cfg)

    def forward(
        self,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
    ) -> torch.Tensor:
        self.unet.train()

        t_view = t.view(x_1.shape[0], 1, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1

        drop_mask = torch.rand((x_1.shape[0],), device=x_1.device) < self.cfg.p_uncond
        return self.unet(x_t, t, context_acts, context_obs, drop_mask)

    @torch.no_grad()
    def sample(
        self,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
        num_ts: int,
        guidance_scale: Optional[float],
        img_wh: tuple[int, int],
    ) -> torch.Tensor:
        self.unet.eval()
        B = context_acts.shape[0]
        H, W = img_wh

        x_t = torch.randn(B, self.cfg.input_channels, H, W, device=context_acts.device)

        for t in torch.linspace(0, 1, num_ts, device=context_acts.device):
            t = t.unsqueeze(0).expand(B, 1)
            drop_cond_mask = torch.ones(
                (B,), device=context_acts.device, dtype=torch.bool
            )
            if guidance_scale is None:
                u_t = self.unet(x_t, t, context_acts, context_obs, ~drop_cond_mask)
            else:
                u_t_uncond = self.unet(
                    x_t, t, context_acts, context_obs, drop_cond_mask
                )
                u_t_cond = self.unet(x_t, t, context_acts, context_obs, ~drop_cond_mask)
                u_t = u_t_uncond + guidance_scale * (u_t_cond - u_t_uncond)

            x_t = x_t + (1 / num_ts) * u_t

        return x_t
