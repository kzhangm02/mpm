import os
import sys
import yaml
import torch
import argparse
from munch import munchify
from model import LatentDiffusion
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('medium')

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(idx):
    torch.manual_seed(idx)
    torch.cuda.manual_seed(idx)

def main(config_filepath, random_seed):
    seed(random_seed)
    seed_everything(random_seed)
    cfg = load_config(config_filepath)
    cfg = munchify(cfg)
    
    log_dir = cfg.log_dir + f'/seed={random_seed}'
    model = LatentDiffusion(
        log_dir=log_dir,
        seed=random_seed,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        lr_schedule=cfg.lr_schedule,
        data_path=cfg.data_path,
    )
    model.train_vae()

    best_vae_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/checkpoints/',
        filename='vae-epoch={epoch}-val_loss={val/loss:.5f}',
        auto_insert_metric_name=False,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )
    trainer = Trainer(
        accelerator='gpu',
        num_nodes=1,
        strategy='auto',
        deterministic=True,
        default_root_dir=log_dir,
        max_epochs=cfg.num_epochs,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        callbacks=[best_vae_checkpoint_callback],
        detect_anomaly=False,
    )
    trainer.fit(model)

    def load_vae_ckpt(ckpt):
        ckpt = torch.load(ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]
        vae_state_dict = {
            k.replace("vae.", ""): v 
            for k, v in state_dict.items()
            if k.startswith("vae.")
        }
        model.vae.load_state_dict(vae_state_dict, strict=False)

    vae_ckpt = best_vae_checkpoint_callback.best_model_path
    load_vae_ckpt(vae_ckpt)
    model.train_unet()
    model.compute_latent_stats()

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/checkpoints/',
        filename='epoch={epoch}-val_loss={val/loss:.5f}',
        auto_insert_metric_name=False,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )
    trainer = Trainer(
        accelerator='gpu',
        num_nodes=1,
        strategy='auto',
        deterministic=True,
        default_root_dir=log_dir,
        max_epochs=cfg.num_epochs,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback],
        detect_anomaly=False,
    )
    trainer.fit(model)

    def load_eval_ckpt(ckpt):
        ckpt = torch.load(ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)

    eval_ckpt_path = best_checkpoint_callback.best_model_path
    load_eval_ckpt(eval_ckpt_path)
    model.eval()
    model.freeze()
    trainer = Trainer(
        accelerator='gpu',
        num_nodes=1,
        strategy='auto',
        deterministic=True,
        default_root_dir=log_dir,
        val_check_interval=1.0,
    )

    trainer.test(model)
    model.save_slots()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed',
    )
    args = parser.parse_args()
    main(args.config, args.seed)