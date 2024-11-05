# train_pl.py
import torch
torch.set_float32_matmul_precision('medium')
from importlib.resources import files
from argparse import ArgumentParser
import sys
import os
# Add the parent directory of f5_tts to Python path
sys.path.append(os.path.abspath("/root/workdir/skrivtesnakk/F5-TTS/src/"))
import lightning as L
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from f5_tts.model import CFM, DiT, UNetT
from f5_tts.model.dataset import load_dataset, DynamicBatchSampler, collate_fn
from f5_tts.model.utils import get_tokenizer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'

tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
dataset_name = "podcast"

# -------------------------- Training Settings -------------------------- #

exp_name = "F5-mini" #"F5TTS_Base"  # F5TTS_Base | E2TTS_Base

learning_rate = 7.5e-5

batch_size_per_gpu = 38400  # Assuming frames per batch per GPU
batch_size_type = "frame"  # "frame" or "sample"
max_samples = 64  # max sequences per batch if use frame-wise batch_size
grad_accumulation_steps = 1  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 1000  # use linear decay, thus epochs control the slope
num_warmup_updates = 10000  # warmup steps
save_per_updates = 10000  # save checkpoint per steps
last_per_steps = 20000  # save last checkpoint per steps

# Model params
if exp_name == "F5TTS_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
elif exp_name == "F5-mini":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)

elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


# ------------------------ Lightning Module ------------------------------#


class LitCFMModel(L.LightningModule):
    def __init__(self, model, learning_rate, num_warmup_updates, total_steps, max_grad_norm=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_warmup_updates = num_warmup_updates
        self.total_steps = total_steps
        self.max_grad_norm = max_grad_norm
        self.save_hyperparameters()  # Saves hyperparameters for logging

    def forward(self, mel_spec, text_inputs, mel_lengths):
        return self.model(mel_spec, text=text_inputs, lens=mel_lengths)
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, phase="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase="val")
    
    def step(self, batch, phase):
        text_inputs = batch["text"]
        mel_spec = batch["mel"].permute(0, 2, 1)
        mel_lengths = batch["mel_lengths"]

        loss, cond, pred = self.model(
            mel_spec, text=text_inputs, lens=mel_lengths
        )
        self.log(f'{phase}/loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.num_warmup_updates
        decay_steps = self.total_steps - warmup_steps
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency":1}
        return [optimizer], [scheduler]


# --------------------------- Main Training Script ------------------------------#

def main():
    # Set seeds for reproducibility
    L.seed_everything(666)
    
    # Tokenizer
    if tokenizer == "custom":
        effective_tokenizer_path = tokenizer_path
    else:
        effective_tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(effective_tokenizer_path, tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    train_dataset, val_dataset = load_dataset(dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    data_sets = {'train' : train_dataset,'val' : val_dataset}

    # Create DataLoader with DynamicBatchSampler if batch_size_type is 'frame'
    if batch_size_type == "frame":
        dataloaders = {}
        for phase, ds in data_sets.items():
            sampler = SequentialSampler(ds)
            batch_sampler = DynamicBatchSampler(
                sampler, 
                frames_threshold=batch_size_per_gpu, 
                max_samples=max_samples, 
                random_seed=666, 
                drop_last=False
            )
            dl = DataLoader(
                ds,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
            dataloaders[phase] = dl
    # elif batch_size_type == "sample":
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         collate_fn=collate_fn,
    #         num_workers=8,
    #         pin_memory=True,
    #         persistent_workers=True,
    #         batch_size=batch_size_per_gpu,
    #         shuffle=True,
    #     )
    else:
        raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {batch_size_type}")

    total_steps = (len(dataloaders['train']) * epochs) // grad_accumulation_steps

    lit_model = LitCFMModel(model, learning_rate, num_warmup_updates, total_steps, max_grad_norm)

    # Initialize WandbLogger if wandb is used
    wandb_logger = WandbLogger(project='CFM-TTS', name=exp_name, id=wandb_resume_id, resume="allow")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(files("f5_tts").joinpath(f"../../ckpts/{exp_name}")),
        save_top_k=-1,
        every_n_train_steps=save_per_updates,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        gradient_clip_val=max_grad_norm,
        accumulate_grad_batches=grad_accumulation_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        # You can specify devices, accelerator, etc., depending on your setup
        devices="auto",
        accelerator="auto",
        precision="bf16",
        val_check_interval = 2000,
        #limit_train_batches=5,
        # Replace 'auto' with the number of GPUs if you want to specify
    )

    trainer.fit(lit_model, dataloaders['train'], dataloaders['val'])


if __name__ == "__main__":
    main()
