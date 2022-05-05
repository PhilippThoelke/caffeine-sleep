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

        # input projection layer
        self.window_length = self.hparams.epoch_length // self.hparams.num_tokens
        self.proj = nn.Linear(self.window_length, self.hparams.embedding_dim)
        # transformer encoder
        self.pe = PositionalEncoding(
            self.hparams.embedding_dim, dropout=self.hparams.dropout
        )
        self.register_parameter(
            "class_token", nn.Parameter(torch.randn(self.hparams.embedding_dim))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            self.hparams.embedding_dim,
            nhead=8,
            dim_feedforward=self.hparams.embedding_dim * 2,
            dropout=self.hparams.dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.hparams.num_layers)
        # output network
        self.outnet = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        if self.hparams.subject_penalty > 0:
            assert (
                num_subjects is not None
            ), "num_subjects can't be None if subject_penalty > 0"

            # subject identifier for regularization
            self.subject_identifier = nn.Linear(
                self.hparams.embedding_dim, num_subjects
            )

    def forward(self, x, return_class_token=False, return_logits=False):
        # add a batch dimension if required
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # crop sequence to be divisible by the desired number of tokens
        x = x[:, : self.hparams.num_tokens * self.window_length]
        # reshape x from (B x time x elec) to (token x B x window_length)
        x = x.view(x.size(0), self.hparams.num_tokens, self.window_length, x.size(2))
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), -1, self.window_length)
        x = x.permute(1, 0, 2)
        # standardize data
        x = (x - self.mean) / self.std
        # linear projection into embedding dimension
        x = self.proj(x)
        # add positional encoding
        x = self.pe(x)
        # prepend class token to the sequence
        x = torch.cat([self.class_token[None, None].repeat(1, x.size(1), 1), x], dim=0)
        # pass sequence through the transformer and extract class tokens
        x = self.encoder(x)[0]
        # apply output model
        y = self.outnet(x).squeeze()
        if not return_logits:
            y = torch.sigmoid(y)
        if return_class_token:
            return y, x
        return y

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="val")

    def step(self, batch, batch_idx, running_stage):
        x, condition, stage, subject = batch
        pred, rep = self(x, return_class_token=True, return_logits=True)

        # loss
        loss = F.binary_cross_entropy_with_logits(pred, condition.float())
        self.log(f"{running_stage}_loss", loss)

        # accuracy
        acc = ((torch.sigmoid(pred) > 0.5).int() == condition).float().mean()
        self.log(f"{running_stage}_acc", acc)

        # penalize identification of subjects
        if self.hparams.subject_penalty > 0:
            subject_pred = self.subject_identifier(rep)
            subject_loss = F.cross_entropy(subject_pred, subject)
            self.log(f"{running_stage}_subject_loss", acc)
            loss = loss - subject_loss * self.hparams.subject_penalty

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.7, patience=5
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor="train_loss")

    def training_epoch_end(self, *args, **kwargs):
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return super().training_epoch_end(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs):
        if self.global_step < self.hparams.warmup_steps:
            self.optimizers().param_groups[0]["lr"] = self.hparams.learning_rate * (
                (self.global_step + 1) / self.hparams.warmup_steps
            )
        return super().optimizer_step(*args, **kwargs)


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
