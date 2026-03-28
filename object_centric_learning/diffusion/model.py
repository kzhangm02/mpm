import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
from dataset import TetrominoesDataset
from torchvision.utils import save_image
from model_utils import VAE, UNet, GaussianDiffusion, GaussianDiffusion_v2
from diffusers import UNet2DConditionModel

latent_ch = 8

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    
class LatentDiffusion(pl.LightningModule):

    def __init__(
        self,
        seed: int=42,
        num_slots: int=4,
        learning_rate: float=1e-4,
        weight_decay: float=1e-2,
        log_dir: str='logs',
        train_batch: int=512,
        val_batch: int=256,
        test_batch: int=256,
        num_workers: int=4,
        lr_schedule: list=[],
        data_path: str='tetrominoes.npz',
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_dict = {
            'train': self.hparams.train_batch,
            'val': self.hparams.val_batch,
            'test': self.hparams.test_batch,
        }
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True}
        self.vae_pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions/vae')
        self.diff_pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions/diffusion')
        mkdir(self.vae_pred_log_dir)
        mkdir(self.diff_pred_log_dir)

        self.T = 1000
        self.beta = 0.01
        self.trained_vae = False
        self.vae = VAE(num_slots, 64)
        self.unet = UNet(cond_c=8*latent_ch)
        self.diffusion = GaussianDiffusion(self.T)
        self.vae_objects = []
        self.vae_alphas = []
        self.diff_objects = []
        self.diff_alphas = []
        self.imgs = []
    
    def train_vae(self):
        print('Training VAE')
        self.training_vae = True
        self.training_unet = False
    
    def train_unet(self):
        print('Training UNet')
        self.training_vae = False
        self.training_unet = True
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
    
    def compute_latent_stats(self):
        latents = []
        self.setup('fit')
        dataloader = self.train_dataloader()
        self.vae.to('cuda:0')
        for batch in dataloader:
            image = (batch['image'].cuda() / 255).permute(0,3,1,2)
            _, _, _, mu, logvar, z = self.vae(image)
            latents.append(mu)
        latents = torch.cat(latents, dim=0)
        self.z_min = torch.amin(latents, dim=0)
        self.z_max = torch.amax(latents, dim=0)
        self.z_range = self.z_max - self.z_min + 1e-8
        
    def training_step(self, batch, batch_idx, split='train'):
        if self.training_vae:
            image = (batch['image'].cuda() / 255).permute(0,3,1,2)
            combined, _, _, mu, logvar, z = self.vae(image)
            rec_loss = F.mse_loss(combined, image)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + self.beta * kl_loss
            self.mylog(split, 'MSE', rec_loss)
            self.mylog(split, 'KL', kl_loss)
            self.mylog(split, 'loss', loss)
        elif self.training_unet:
            image = (batch['image'].cuda() / 255).permute(0,3,1,2)
            mu, logvar, z = self.vae.encode(image)
            B = z.size(0)
            mu = mu.reshape(B,4*latent_ch,8,8)
            logvar = logvar.reshape(B,4*latent_ch,8,8)
            c = torch.cat([mu, logvar], dim=1)
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=z.device)
            zt, eps = self.diffusion.sample_from_forward_process(mu, t)
            pred_eps = self.unet(zt, c, t)
            loss = F.mse_loss(pred_eps, eps)
            self.mylog(split, 'loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, split='val'):
        if self.training_vae:
            image = (batch['image'].cuda() / 255).permute(0,3,1,2)
            combined, _, _, mu, logvar, z = self.vae(image)
            rec_loss = F.mse_loss(combined, image)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + self.beta * kl_loss
            self.mylog(split, 'MSE', rec_loss)
            self.mylog(split, 'KL', kl_loss)
            self.mylog(split, 'loss', loss)
        elif self.training_unet:
            image = (batch['image'].cuda() / 255).permute(0,3,1,2)
            mu, logvar, z = self.vae.encode(image)
            B = z.size(0)
            mu = mu.reshape(B,4*latent_ch,8,8)
            logvar = logvar.reshape(B,4*latent_ch,8,8)
            c = torch.cat([mu, logvar], dim=1)
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=z.device)
            zt, eps = self.diffusion.sample_from_forward_process(mu, t)
            pred_eps = self.unet(zt, c, t)
            loss = F.mse_loss(pred_eps, eps)
            self.mylog(split, 'loss', loss)
        return loss

    def test_step(self, batch, batch_idx, split='test'):
        image = (batch['image'].cuda() / 255).permute(0,3,1,2)
        vae_combined, vae_recons, vae_alphas, mu, logvar, z = self.vae(image)
        rec_loss = F.mse_loss(vae_combined, image)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = rec_loss + self.beta * kl_loss

        B = z.size(0)
        mu = mu.reshape(B,4*latent_ch,8,8)
        logvar = logvar.reshape(B,4*latent_ch,8,8)
        c = torch.cat([mu, logvar], dim=1)
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=z.device)
        zt, eps = self.diffusion.sample_from_forward_process(mu, t)
        pred_eps = self.unet(zt, c, t)
        unet_loss = F.mse_loss(pred_eps, eps)

        diff_zt = torch.randn((B, 4*latent_ch, 8, 8), device=zt.device)
        diff_z = self.diffusion.sample_from_reverse_process(
            self.unet, diff_zt, timesteps=self.T,
            model_kwargs={'c': c},
        )
        diff_combined, diff_recons, diff_alphas = self.vae.decode(diff_z)
        diff_loss = F.mse_loss(diff_combined, image)

        vae_save_path = os.path.join(self.vae_pred_log_dir, f'{batch_idx}.png')
        self.save(image[0], vae_combined[0], vae_recons[0], vae_save_path)
        diff_save_path = os.path.join(self.diff_pred_log_dir, f'{batch_idx}.png')
        self.save(image[0], diff_combined[0], diff_recons[0], diff_save_path)

        self.vae_objects.append(vae_recons.detach().cpu().numpy())
        self.vae_alphas.append(vae_alphas.detach().cpu().numpy())
        self.diff_objects.append(diff_recons.detach().cpu().numpy())
        self.diff_alphas.append(diff_alphas.detach().cpu().numpy())
        self.imgs.append(image.detach().cpu().numpy())
        
        self.mylog(split, 'MSE', rec_loss)
        self.mylog(split, 'KL', kl_loss)
        self.mylog(split, 'vae_loss', vae_loss)
        self.mylog(split, 'unet_loss', unet_loss)
        self.mylog(split, 'diffusion_MSE', diff_loss)
        return diff_loss

    def save(self, image, combined, recons, save_path):
        comparison = torch.cat([
            F.pad(image, (2,2,2,2), value=1), 
            F.pad(combined, (2,2,2,2), value=1),
            *F.pad(recons, (2,2,2,2), value=1)
        ], dim=2)
        save_image(comparison.cpu(), save_path, nrow=1)
    
    def save_slots(self):
        self.vae_objects = np.concatenate([*self.vae_objects], axis=0)
        self.vae_objects = np.reshape(self.vae_objects, (-1, 3, 35, 35))
        self.vae_alphas = np.concatenate([*self.vae_alphas], axis=0)
        self.vae_alphas = np.reshape(self.vae_alphas, (-1, 4, 35, 35))
        self.diff_objects = np.concatenate([*self.diff_objects], axis=0)
        self.diff_objects = np.reshape(self.diff_objects, (-1, 3, 35, 35))
        self.diff_alphas = np.concatenate([*self.diff_alphas], axis=0)
        self.diff_alphas = np.reshape(self.diff_alphas, (-1, 4, 35, 35))
        self.imgs = np.concatenate([*self.imgs], axis=0)
        self.imgs = np.reshape(self.imgs, (-1, 3, 35, 35))
        vae_pred_log_dir = self.vae_pred_log_dir
        diff_pred_log_dir = self.diff_pred_log_dir
        
        np.save(os.path.join(vae_pred_log_dir, f'objects.npy'), self.vae_objects)
        np.save(os.path.join(vae_pred_log_dir, f'alphas.npy'), self.vae_alphas)
        np.save(os.path.join(vae_pred_log_dir, f'images.npy'), self.imgs)

        np.save(os.path.join(diff_pred_log_dir, f'objects.npy'), self.diff_objects)
        np.save(os.path.join(diff_pred_log_dir, f'alphas.npy'), self.diff_alphas)
        np.save(os.path.join(diff_pred_log_dir, f'images.npy'), self.imgs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.5,
            milestones=self.hparams.lr_schedule, 
        )
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = TetrominoesDataset(split='train', 
                fp=self.hparams.data_path, seed=self.hparams.seed)
            self.val_dataset = TetrominoesDataset(split='val', 
                fp=self.hparams.data_path, seed=self.hparams.seed)
        if stage == 'test':
            self.test_dataset = TetrominoesDataset(split='test', 
                fp=self.hparams.data_path, seed=self.hparams.seed)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch,
            shuffle=True,
            **self.kwargs,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.val_batch,
            shuffle=False,
            **self.kwargs,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.test_batch,
            shuffle=False,
            **self.kwargs,
        )
        return test_loader
    
    def mylog(self, phase, name, value):
        batch_size = self.batch_size_dict[phase]
        self.log(
            f'{phase}/{name}', value, on_epoch=True, prog_bar=True, 
            logger=True, batch_size=batch_size, sync_dist=True,
        )