import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, UNet2DModel, DDPMScheduler, DDIMScheduler

unsqueeze3x = lambda x: x[..., None, None, None]
latent_ch = 8

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.block(x)


class VAE(nn.Module):
    def __init__(self, num_slots, slot_size=64):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.enc = nn.Sequential(
            ConvBlock(3, 32),
            nn.Conv2d(32, 64, 4, 2, 1),    # 35 -> 17
            nn.SiLU(),
            ConvBlock(64, 64),
            nn.Conv2d(64, 128, 4, 2, 1),   # 17 -> 8
            nn.SiLU(),
            ConvBlock(128, 128),
            nn.Conv2d(128, 8 * latent_ch, 1),
        )
        self.flat_dim = 8 * latent_ch * 8 * 8
        self.dec = nn.Sequential(
            ConvBlock(latent_ch, 128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1),  # 8 -> 17
            nn.SiLU(),
            ConvBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=1),   # 18 -> 35
            nn.SiLU(),
            ConvBlock(32, 32),
            nn.Conv2d(32, 4, 3, padding=1),       # output 4 channels
        )

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        combined, recons, alphas = self.decode(z)
        return combined, recons, alphas, mu, logvar, z
    
    def encode(self, x):
        B = x.size(0)
        K = self.num_slots
        h = self.enc(x)                     # (B, 2L, 8, 8)
        mu = h[:, :4*latent_ch]
        logvar = h[:, 4*latent_ch:]
        z = self.reparam(mu, logvar)        # (B, 256)
        z = z.view(B, 4*latent_ch, 8, 8)    # (B, 4, 8, 8)
        return mu, logvar, z
    
    def decode(self, z):
        B = z.size(0)
        K = self.num_slots
        z = z = z.view(4*B, latent_ch, 8, 8)
        y = self.dec(z)                     # (4B, 4, 35, 35)
        y = y.view(B, K, 4, 35, 35)
        combined, recons, alphas = self.merge_slots(y)
        return combined, recons, alphas
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def merge_slots(self, x):
        recons = F.hardtanh(x[:, :, :3], 0, 1)
        alphas = F.softmax(x[:, :, -1:], dim=1)
        recons = recons * alphas
        combined = recons.sum(dim=1)
        return combined, recons, alphas


class ImageCondEncoder(nn.Module):
    def __init__(self, cond_c=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32),
            nn.Conv2d(32, 64, 4, 2, 1),      # 35 -> 17
            nn.SiLU(),
            ConvBlock(64, 64),
            nn.Conv2d(64, cond_c, 4, 2, 1),  # 17 -> 8
            nn.SiLU(),
            ConvBlock(cond_c, cond_c),
        )

    def forward(self, x):
        return self.net(x)   # (B, cond_channels, 8, 8)


class UNet(nn.Module):

    def __init__(self, cond_c=128):
        super().__init__()
        self.cond_c = cond_c
        self.unet = UNet2DModel(
            in_channels=4*latent_ch+cond_c,   # noise + spatial conditioning
            out_channels=4*latent_ch,
            sample_size=8,
            block_out_channels=(64, 128),
            down_block_types=("AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D"),
        )

    def forward(self, noise_latent, c, t):
        x = torch.cat([noise_latent, c], 1)    # (B, 4+cond_c,8,8)
        out = self.unet(
            sample=x,
            timestep=t,
        ).sample
        return out
    

class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=100, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: np.cos((t / self.timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars["alpha_bar"][t].sqrt())
                * (xt - unsqueeze3x((1 - scalars["alpha_bar"][t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars["alpha_bar"][t].sqrt() * scalars["beta"][t])
                / ((1 - scalars["alpha_bar"][t]) * scalars["alpha"][t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars["alpha"][t] - scalars["alpha_bar"][t])
                / ((1 - scalars["alpha_bar"][t]) * scalars["alpha"][t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return dict([(k, v.float()) for (k, v) in all_scalars.items()])

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars["alpha_bar"][t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars["alpha_bar"][t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, t=current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars["beta_tilde"][current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final
    
class GaussianDiffusion_v2:
    def __init__(self, timesteps=100, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.ddpm_train = DDPMScheduler(
            num_train_timesteps=timesteps,
            clip_sample=False,
        )

    def sample_from_forward_process(self, x0, t):
        eps = torch.randn_like(x0)
        xt = self.ddpm_train.add_noise(x0, eps, t)
        return xt, eps

    def predict_x0_from_xt_eps(self, xt, eps, t):
        a_bar = self.ddpm_train.alphas_cumprod.to(xt.device)[t].view(-1, 1, 1, 1)
        return (xt - (1.0 - a_bar).sqrt() * eps) / a_bar.sqrt()

    def check_one_step_inversion(self, B=10, C=4, H=8, W=8):
        x0 = torch.randn(B, C, H, W, device=self.device, dtype=torch.float32)
        t  = torch.full((B,), 1, device=self.device, dtype=torch.long)
        xt, eps = self.sample_from_forward_process(x0, t)        
        sched = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
        sched.set_timesteps(1)
        sched.eta = 0.0
        x0_hat = sched.step(eps, 1, xt).prev_sample
        mse = ((x0 - x0_hat) ** 2).mean()
        max_abs = (x0 - x0_hat).abs().max()
        print("one-step mse:", float(mse), "max_abs:", float(max_abs))
        return mse, max_abs

    def sample_from_reverse_process(
            self, model, xT, num_inference_steps=None, 
            model_kwargs=None, sampler="ddpm", eta=0.0
        ):
        model_kwargs = model_kwargs or {}
        num_inference_steps = num_inference_steps or self.timesteps
        if sampler == "ddpm":
            sched = DDPMScheduler(num_train_timesteps=self.timesteps, clip_sample=False)
        elif sampler == "ddim":
            sched = DDIMScheduler(num_train_timesteps=self.timesteps, clip_sample=False)
            sched.eta = 0
        else:
            raise ValueError("sampler must be 'ddpm' or 'ddim'")
        sched.set_timesteps(num_inference_steps, device=xT.device)

        x = xT
        for t in sched.timesteps:  # IMPORTANT: goes high -> low
            t_batch = torch.full((x.size(0),), int(t), device=x.device, dtype=torch.long)
            eps = model(x, t=t_batch, **model_kwargs)
            x = sched.step(eps, t, x).prev_sample
        return x

    def check_oracle_chain(self, B=10, C=4, H=8, W=8, num_inference_steps=None, sampler="ddim", eta=0.0):
        num_inference_steps = num_inference_steps or self.timesteps
        if sampler == "ddpm":
            sched = DDPMScheduler(num_train_timesteps=self.timesteps, clip_sample=False)
        elif sampler == "ddim":
            sched = DDIMScheduler(num_train_timesteps=self.timesteps, clip_sample=False)
            sched.eta = 0
        else:
            raise ValueError("sampler must be 'ddpm' or 'ddim'")

        sched.set_timesteps(num_inference_steps, device=self.device)
        x0 = torch.randn(B, C, H, W, device=self.device, dtype=torch.float32)
        t_last = int(sched.timesteps[0])
        t_last_batch = torch.full((B,), t_last, device=self.device, dtype=torch.long)
        eps0 = torch.randn_like(x0)
        x = self.ddpm_train.add_noise(x0, eps0, t_last_batch)
        for t in sched.timesteps:
            t_int = int(t)
            t_batch = torch.full((B,), t_int, device=self.device, dtype=torch.long)
            a_bar = self.ddpm_train.alphas_cumprod.to(self.device)[t_batch].view(-1, 1, 1, 1)
            eps_oracle = (x - a_bar.sqrt() * x0) / (1.0 - a_bar).sqrt()
            x = sched.step(eps_oracle, t, x).prev_sample
        mse = ((x0 - x) ** 2).mean()
        print(f"oracle full-chain mse ({sampler}, eta={eta}):", float(mse))
        return mse