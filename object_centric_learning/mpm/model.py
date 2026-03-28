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
from model_utils import SlotAttentionAutoencoder

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    
class SlotAutoencoder(pl.LightningModule):

    def __init__(
        self,
        seed: int=42,
        beta: float=1.0,
        num_slots: int=5,
        num_iterations: int=3,
        num_slot_clusters: int=100,
        additive_decoder: bool=False,
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
        self.pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions')
        mkdir(self.pred_log_dir)

        self.resolution = (35, 35, 3)
        if additive_decoder:
            self.loss = torch.nn.MSELoss()

        self.model = SlotAttentionAutoencoder(
            self.resolution, num_slots, num_iterations,
            num_slot_clusters, additive_decoder,
        ).cuda()
        self.objects = []
        self.slots = []
        self.alphas = []
        self.imgs = []
        if num_slot_clusters is not None and num_slot_clusters > 0:
            self.global_clustering = True
            self.mixing = []
        else:
            self.global_clustering = False

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        image = batch['image'].cuda() / 255
        combined, recons, _, _, attns = self(image)
        qz = attns[0].permute(0,2,1)
        qz = qz.reshape(-1, 4, 35, 35, 1)
        if self.hparams.additive_decoder:
            train_recon = self.loss(image, combined)
        else:
            train_mse = (image.unsqueeze(1) - recons).square()
            train_recon = 0.5 * (qz * train_mse).sum(dim=(1,4)).mean()
        train_H = (qz * torch.log(qz)).sum(dim=1).mean()
        train_loss = train_recon + self.hparams.beta * train_H
        self.mylog('train', 'loss', train_loss)
        self.mylog('train', 'recon', train_recon)
        self.mylog('train', 'H', train_H)
        return train_loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'].cuda() /  255
        combined, recons, _, _, attns, = self(image)
        qz = attns[0].permute(0,2,1)
        qz = qz.reshape(-1, 4, 35, 35, 1)
        if self.hparams.additive_decoder:
            val_recon = self.loss(image, combined)
        else:
            val_mse = (image.unsqueeze(1) - recons).square()
            val_recon = 0.5 * (qz * val_mse).sum(dim=(1,4)).mean()
        val_H = (qz * torch.log(qz)).sum(dim=1).mean()
        val_loss = val_recon + self.hparams.beta * val_H
        self.mylog('val', 'loss', val_loss)
        self.mylog('val', 'recon', val_recon)
        self.mylog('val', 'H', val_H)
        return val_loss

    def test_step(self, batch, batch_idx):
        image = batch['image'].cuda() /  255
        combined, recons, masks, slots, attns = self(image)
        qz = attns[0].permute(0,2,1)
        qz = qz.reshape(-1, 4, 35, 35, 1)
        if self.hparams.additive_decoder:
            test_recon = self.loss(image, combined)
            self.objects.append((recons * masks).detach().cpu().numpy())
            self.alphas.append(masks.detach().cpu().numpy())
        else:
            test_mse = (image.unsqueeze(1) - recons).square()
            test_recon = 0.5 * (qz * test_mse).sum(dim=(1,4)).mean()
            self.objects.append((recons * qz).detach().cpu().numpy())
            self.alphas.append(qz.detach().cpu().numpy())
            
        save_path = os.path.join(self.pred_log_dir, f'{batch_idx}.png')
        self.save(image, combined, recons, masks, attns, save_path)
        self.slots.append(slots.detach().cpu().numpy())
        self.imgs.append(image.detach().cpu().numpy())
        if self.global_clustering:
            self.mixing.append(attns[1].detach().cpu().numpy())

        test_H = (qz * torch.log(qz)).sum(dim=1).mean()
        test_loss = test_recon + self.hparams.beta * test_H
        self.mylog('test', 'loss', test_loss)
        self.mylog('test', 'recon', test_recon)
        self.mylog('test', 'H', test_H)
        return test_loss

    def save(self, image, combined, recons, masks, attns, save_path):
        if self.hparams.additive_decoder:
            masked_recons = recons * masks
            masked_recons = torch.permute(masked_recons, (1, 0, 2, 3, 4))
            comparison = torch.cat([
                F.pad(image, (0,0,2,2,2,2), value=1), 
                F.pad(combined, (0,0,2,2,2,2), value=1),
                *F.pad(masked_recons, (0,0,2,2,2,2), value=1)
            ], dim=1)
        else:
            qz = attns[0].permute(0,2,1)
            qz = qz.reshape(-1, 4, 35, 35, 1)
            masked_recons = recons * qz
            masked_recons = masked_recons.permute(1, 0, 2, 3, 4)
            comparison = comparison = torch.cat([
                F.pad(image, (0,0,2,2,2,2), value=1),
                *F.pad(masked_recons, (0,0,2,2,2,2), value=1)
            ], dim=1)
        comparison = torch.permute(comparison, (0, 3, 2, 1))
        save_image(comparison[0].cpu(), save_path, nrow=1)
    
    def save_slots(self):
        self.objects = np.concatenate([*self.objects], axis=0)
        self.objects = np.reshape(self.objects, (-1, *self.resolution))
        self.slots = np.concatenate([*self.slots], axis=0)
        self.slots = np.reshape(self.slots, (-1, self.slots.shape[-1]))
        self.alphas = np.concatenate([*self.alphas], axis=0)
        self.alphas = np.reshape(self.alphas, (-1, 4, 35, 35, 1))
        self.imgs = np.concatenate([*self.imgs], axis=0)
        self.imgs = np.reshape(self.imgs, (-1, 35, 35, 3))
        np.save(os.path.join(self.pred_log_dir, f'objects.npy'), self.objects)
        np.save(os.path.join(self.pred_log_dir, f'slots.npy'), self.slots)
        np.save(os.path.join(self.pred_log_dir, f'alphas.npy'), self.alphas)
        np.save(os.path.join(self.pred_log_dir, f'images.npy'), self.imgs)
        if self.global_clustering:
            self.mixing = np.concatenate([*self.mixing], axis=0)
            np.save(os.path.join(self.pred_log_dir, f'mixing.npy'), self.mixing)
    
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