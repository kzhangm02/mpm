
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
from transformers import BertModel
from torchvision.utils import save_image


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


class TextSlotAttentionAutoencoder(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        slot_size,
        num_slots,
        num_iterations,
        num_slot_clusters,  
    ):
        super(TextSlotAttentionAutoencoder, self).__init__()
        self.tokenizer = tokenizer
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_slot_clusters = num_slot_clusters
        
        if self.tokenizer == 'bert':
            self.hidden_dim = 768
            self.vocab_size = 28996
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
            freeze_bert = True
            if freeze_bert:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"{self.tokenizer} not implemented")
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(768, 768), torch.nn.ReLU(),
            torch.nn.Linear(768, 2 * 768), torch.nn.ReLU(),
            torch.nn.Linear(2 * 768, self.vocab_size),
        )
        self.N = 512
        self.pos_enc = torch.nn.Parameter(
            torch.randn(1, self.N, 768)
        )

    def forward(self, x, masks, word_starts):
        batch_size = x.shape[0]
        x = self.encoder(x, attention_mask=masks)
        z = x.last_hidden_state + self.pos_enc
        logits = self.decoder(z)
        return logits
    

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
        self.loss = torch.nn.CrossEntropyLoss()
        
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
        logits = self(tokens, masks, word_starts)
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijk, ijk -> ij', logits, onehot_tokens)
        loss = (nll * masks).sum(dim=1).mean()
        self.mylog('train', 'loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masks = batch['masks']
        tokens = batch['input_ids']
        word_starts = batch['word_starts']
        logits = self(tokens, masks, word_starts)
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijk, ijk -> ij', logits, onehot_tokens)
        loss = (nll * masks).sum(dim=1).mean()
        self.mylog('val', 'loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        masks = batch['masks']
        tokens = batch['input_ids']
        word_starts = batch['word_starts']
        logits = self(tokens, masks, word_starts)
        onehot_tokens = F.one_hot(tokens, num_classes=self.vocab_size).float()
        nll = -torch.einsum('ijk, ijk -> ij', logits, onehot_tokens)
        loss = (nll * masks).sum(dim=1).mean()
        self.mylog('test', 'loss', loss)
        return loss
    
    def save_results(self):
        pass
    
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