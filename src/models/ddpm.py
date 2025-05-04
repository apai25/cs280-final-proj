from typing import Optional, Tuple

import torch
from torch import nn

from configs.model_config import ModelConfig
from src.models.unet import UNet


class DDPM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(DDPM, self).__init__()
        self.cfg = cfg
        self.unet = UNet(cfg)

        ddpm_schedule = self._ddpm_schedule()
        self.register_buffer("ddpm_betas", ddpm_schedule["ddpm_betas"])
        self.register_buffer("ddpm_alphas", ddpm_schedule["ddpm_alphas"])
        self.register_buffer("ddpm_alpha_bars", ddpm_schedule["ddpm_alpha_bars"])

    def forward(
        self,
        x_0: torch.Tensor,
        t_int: torch.Tensor,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.unet.train()

        eps = torch.randn_like(x_0)
        alpha_bar_t = self.ddpm_alpha_bars[t_int].view(x_0.shape[0], 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        t = t_int.float().unsqueeze(1)

        drop_uncond = torch.rand((x_0.shape[0],), device=x_0.device) < self.cfg.p_uncond

        return self.unet(x_t, t, context_acts, context_obs, drop_uncond), eps

    @torch.no_grad()
    def sample(
        self,
        context_acts: torch.Tensor,
        context_obs: torch.Tensor,
        num_steps: int,
        guidance_scale: Optional[float],
        img_wh: tuple[int, int],
    ) -> torch.Tensor:
        self.unet.eval()
        B = context_acts.shape[0]
        H, W = img_wh

        x_t = torch.randn(B, self.cfg.input_channels, H, W, device=context_acts.device)

        for i in reversed(range(num_steps)):
            t_int = torch.full((B,), i, device=context_acts.device)
            t = t_int.float().unsqueeze(1)

            alpha_t = self.ddpm_alphas[i]
            alpha_bar_t = self.ddpm_alpha_bars[i]
            beta_t = self.ddpm_betas[i]
            alpha_bar_t_1 = (
                self.ddpm_alpha_bars[i - 1]
                if i > 0
                else torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
            )

            no_drop = torch.zeros((B,), device=context_acts.device, dtype=torch.bool)
            eps_cond = self.unet(x_t, t, context_acts, context_obs, drop_cond=no_drop)

            if guidance_scale is not None:
                drop = torch.ones((B,), device=context_acts.device, dtype=torch.bool)
                eps_uncond = self.unet(
                    x_t, t, context_acts, context_obs, drop_cond=drop
                )
                eps_pred = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
            else:
                eps_pred = eps_cond

            coef1 = torch.sqrt(alpha_bar_t_1) * beta_t / (1 - alpha_bar_t)
            x_0_hat = (1 / torch.sqrt(alpha_bar_t)) * (
                x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred
            )

            coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_1) / (1 - alpha_bar_t)

            coef3 = torch.sqrt(beta_t)
            z = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)

            x_t = coef1 * x_0_hat + coef2 * x_t + coef3 * z

        return x_t.clamp(-1, 1)

    def _ddpm_schedule(self) -> dict:
        assert self.cfg.beta1 < self.cfg.beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
        betas = torch.linspace(self.cfg.beta1, self.cfg.beta2, self.cfg.num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return {
            "ddpm_betas": betas,
            "ddpm_alphas": alphas,
            "ddpm_alpha_bars": alpha_bars,
        }
