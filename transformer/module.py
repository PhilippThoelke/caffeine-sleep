import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerModule(pl.LightningModule):
    def __init__(self, hparams, mean=0, std=1, num_subjects=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.register_buffer("mean", torch.scalar_tensor(mean))
        self.register_buffer("std", torch.scalar_tensor(std))

        self.sample_length = self.hparams.epoch_length // self.hparams.num_tokens

        # transformer encoder
        self.encoder = EEGEncoder(
            self.hparams.embedding_dim,
            self.hparams.num_layers,
            self.sample_length,
            dropout=self.hparams.dropout,
        )

        # output network
        self.outnet = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        if self.hparams.adversarial_frequency > 0:
            assert (
                num_subjects is not None
            ), "num_subjects can't be None if hparams.adversarial_frequency > 0"

            # subject identifier for regularization
            self.subject_identifier = nn.Sequential(
                nn.Linear(self.hparams.embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_subjects),
            )

    def forward(self, x, return_representation=False, return_logits=False):
        # add a batch dimension if required
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # crop sequence to be divisible by the desired number of tokens
        x = x[:, : self.hparams.num_tokens * self.sample_length]
        # potentially drop some channels
        if len(self.hparams.ignore_channels) > 0:
            ch_mask = torch.ones(x.size(2), dtype=torch.bool)
            ch_mask.scatter_(0, torch.tensor(self.hparams.ignore_channels), False)
            x = x[..., ch_mask]
        # reshape x from (B x time x elec) to (token x B x window_length)
        x = x.view(x.size(0), self.hparams.num_tokens, self.sample_length, x.size(2))
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), -1, self.sample_length)
        x = x.permute(1, 0, 2)
        # standardize data
        x = (x - self.mean) / self.std
        # apply encoder model
        x = self.encoder(x)
        # apply output model
        y = self.outnet(x).squeeze(dim=1)

        if return_logits and return_representation:
            return y, x
        elif not return_logits and return_representation:
            return torch.sigmoid(y), x
        elif return_logits and not return_representation:
            return y
        else:
            return torch.sigmoid(y)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(
            batch, batch_idx, running_stage="train", optimizer_idx=optimizer_idx
        )

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="test")

    def step(self, batch, batch_idx, running_stage, optimizer_idx=None):
        x, condition, stage, subject = batch
        logits, rep = self(x, return_representation=True, return_logits=True)

        if optimizer_idx is None or optimizer_idx == 0:
            # loss
            loss = F.binary_cross_entropy_with_logits(logits, condition.float())
            self.log(f"{running_stage}_loss", loss)

            # accuracy
            acc = ((torch.sigmoid(logits) > 0.5).int() == condition).float().mean()
            self.log(f"{running_stage}_acc", acc)
            return loss
        elif optimizer_idx in [1, 2]:
            # adversarial training subject identification
            subject_logits = self.subject_identifier(rep)
            loss = F.cross_entropy(subject_logits, subject)
            self.log(f"{running_stage}_subject_loss", loss)

            acc = (subject_logits.argmax(dim=1) == subject).float().mean()
            self.log(f"{running_stage}_subject_acc", acc)
            if optimizer_idx == 1:
                # minimize subject loss of the classification network
                return loss
            else:
                # maximize subject loss of the representation network
                return -loss

    def configure_optimizers(self):
        # condition optimizer and scheduler
        condition_opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            condition_opt, factor=0.7, patience=5
        )
        if self.hparams.adversarial_frequency < 1:
            return dict(
                optimizer=condition_opt, lr_scheduler=scheduler, monitor="train_loss"
            )

        # subject classification optimizer
        subject_opt = optim.Adam(
            self.subject_identifier.parameters(), lr=self.hparams.subject_lr
        )
        # adversarial subject optimizer
        adv_opt = optim.Adam(self.encoder.parameters(), lr=self.hparams.adversarial_lr)

        # frequency determines the number of sequential batches of one optimizer
        return (
            dict(
                optimizer=condition_opt,
                lr_scheduler=dict(scheduler=scheduler, monitor="train_loss"),
                frequency=self.hparams.adversarial_frequency,
            ),
            dict(optimizer=subject_opt, frequency=1),
            dict(optimizer=adv_opt, frequency=1),
        )

    def training_epoch_end(self, *args, **kwargs):
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = optimizers[0]
        self.log("lr", opt.param_groups[0]["lr"])
        return super().training_epoch_end(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs):
        if self.global_step < self.hparams.warmup_steps:
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]

            for opt in optimizers:
                opt.param_groups[0]["lr"] = self.hparams.learning_rate * (
                    (self.global_step + 1) / self.hparams.warmup_steps
                )
        return super().optimizer_step(*args, **kwargs)


class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, sample_length, dropout=0.1):
        super().__init__()
        # input projection layer
        self.proj = nn.Linear(sample_length, embedding_dim)
        # positional encoding
        self.pe = PositionalEncoding(embedding_dim, dropout=dropout)
        # class token
        self.register_parameter("class_token", nn.Parameter(torch.randn(embedding_dim)))
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, nn.LayerNorm(embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # linear projection into embedding dimension
        x = self.proj(x)
        # add positional encoding
        x = self.pe(x)
        # prepend class token to the sequence
        x = torch.cat([self.class_token[None, None].repeat(1, x.size(1), 1), x], dim=0)
        # pass sequence through the transformer and extract class tokens
        return self.dropout(self.encoder(x)[0])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
