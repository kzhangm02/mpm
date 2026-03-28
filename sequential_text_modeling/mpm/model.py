import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
from PIL import Image
from dataset import APDataset
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.utils import save_image
from model_utils import TextSlotAttentionAutoencoder

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    
class TextSlotAutoencoder(pl.LightningModule):

    def __init__(
        self,
        seed: int=42,
        beta: float=1.0,
        slot_size: int=64,
        num_slots: int=5,
        num_iterations: int=3,
        num_slot_clusters: int=100,
        learning_rate: float=1e-4,
        weight_decay: float=1e-2,
        log_dir: str='logs',
        train_batch: int=512,
        val_batch: int=256,
        test_batch: int=256,
        num_workers: int=4,
        lr_schedule: list=[],
        encoder: str='bert',
        data_path: str='ap-bert.npz',
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

        self.model = TextSlotAttentionAutoencoder(
            self.hparams.encoder, self.hparams.slot_size,
            num_slots, num_iterations, num_slot_clusters,
        ).cuda()
        self.vocab_size = self.model.vocab_size
        
        self.ids = []
        self.slots = []
        self.attns = []
        self.logits = []
        if num_slot_clusters is not None and num_slot_clusters > 0:
            self.global_clustering = True
            self.mixing = []
        else:
            self.global_clustering = False

    def forward(self, x, masks, word_starts):
        return self.model(x, masks, word_starts)
        
    def training_step(self, batch, batch_idx):
        masks = batch['masks']
        tokens = batch['input_ids']
        word_starts = batch['word_starts']
        logits, _, attns = self(tokens, masks, word_starts)
        normed_logits = F.log_softmax(logits, dim=-1)

        eps = 1e-8
        qz = attns[0]    # qz_shape: [batch_size, seq_len, num_topics]
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijkl, ijl -> ijk', normed_logits, onehot_tokens)
        Eq_nll = (qz * nll).sum(dim=2) * masks
        Ni = masks.sum(dim=1)

        train_recon = (Eq_nll.sum(dim=1) / Ni).mean()
        train_H = ((qz * torch.log(qz.clamp(min=eps))).sum(dim=(1,2)) / Ni).mean()
        train_loss = train_recon + self.hparams.beta * train_H
        self.mylog('train', 'loss', train_loss)
        self.mylog('train', 'recon', train_recon)
        self.mylog('train', 'H', train_H)
        return train_loss

    def validation_step(self, batch, batch_idx):
        masks = batch['masks']
        tokens = batch['input_ids']
        word_starts = batch['word_starts']
        logits, _, attns = self(tokens, masks, word_starts)
        normed_logits = F.log_softmax(logits, dim=-1)

        eps = 1e-8
        qz = attns[0]    # qz_shape: [batch_size, seq_len, num_topics]
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijkl, ijl -> ijk', normed_logits, onehot_tokens)
        Eq_nll = (qz * nll).sum(dim=2) * masks
        Ni = masks.sum(dim=1)

        val_recon = (Eq_nll.sum(dim=1) / Ni).mean()
        val_H = ((qz * torch.log(qz.clamp(min=eps))).sum(dim=(1,2)) / Ni).mean()
        val_loss = val_recon + self.hparams.beta * val_H
        self.mylog('val', 'loss', val_loss)
        self.mylog('val', 'recon', val_recon)
        self.mylog('val', 'H', val_H)
        return val_loss

    def test_step(self, batch, batch_idx):
        ids = batch['doc_ids']
        masks = batch['masks']
        tokens = batch['input_ids']
        word_starts = batch['word_starts']
        logits, slots, attns = self(tokens, masks, word_starts)
        normed_logits = F.log_softmax(logits, dim=-1)

        eps = 1e-8
        qz = attns[0]    # qz_shape: [batch_size, seq_len, num_topics]
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijkl, ijl -> ijk', normed_logits, onehot_tokens)
        Eq_nll = (qz * nll).sum(dim=2) * masks
        Ni = masks.sum(dim=1)

        pred_ll = nll.sum(dim=2) / self.hparams.num_slots
        pred_ll = (pred_ll * masks).sum(dim=1) / Ni
        log_perplex = pred_ll.mean()

        test_recon = (Eq_nll.sum(dim=1) / Ni).mean()
        test_H = ((qz * torch.log(qz.clamp(min=eps))).sum(dim=(1,2)) / Ni).mean()
        test_loss = test_recon + self.hparams.beta * test_H
        self.mylog('test', 'loss', test_loss)
        self.mylog('test', 'recon', test_recon)
        self.mylog('test', 'log_perplexity', log_perplex)
        self.mylog('test', 'H', test_H)
        
        self.ids.append(ids.detach().cpu().numpy())
        self.attns.append(qz.detach().cpu().numpy())
        self.slots.append(slots.detach().cpu().numpy())
        if self.global_clustering:
            mixing = attns[1].reshape(
                (-1, self.hparams.num_slots, self.hparams.num_slot_clusters))
            self.mixing.append(mixing.detach().cpu().numpy())
        return test_loss
    
    def save_results(self):
        self.ids = np.concatenate([*self.ids], axis=0)
        self.attns = np.concatenate([*self.attns], axis=0)
        self.slots = np.concatenate([*self.slots], axis=0)
        np.savez(
            os.path.join(self.pred_log_dir, f'results.npz'), 
            doc_ids=self.ids, 
            attns=self.attns, 
            slots=self.slots,
        )
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
            self.train_dataset = APDataset(split='train', 
                fp=self.hparams.data_path, seed=self.hparams.seed)
            self.val_dataset = APDataset(split='val', 
                fp=self.hparams.data_path, seed=self.hparams.seed)
        if stage == 'test':
            self.test_dataset = APDataset(split='test', 
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