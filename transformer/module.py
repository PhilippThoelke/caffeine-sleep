import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(20, 8), nn.ReLU(), nn.Linear(8, 3))

    def forward(self, x):
        return torch.sigmoid(self.mlp(x.norm(dim=1)))

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, running_stage="val")

    def step(self, batch, batch_idx, running_stage):
        x, condition, stage, subject = batch
        condition_pred, stage_pred, subject_pred = self.mlp(x.norm(dim=1)).T

        # losses
        condition_loss = F.binary_cross_entropy_with_logits(
            condition_pred, condition.float()
        )
        stage_loss = F.binary_cross_entropy_with_logits(stage_pred, stage.float())
        subject_loss = F.binary_cross_entropy_with_logits(subject_pred, subject.float())

        self.log(f"{running_stage}_condition_loss", condition_loss)
        self.log(f"{running_stage}_stage_loss", stage_loss)
        self.log(f"{running_stage}_subject_loss", subject_loss)

        # accuracy
        condition_acc = (
            ((torch.sigmoid(condition_pred) > 0.5).int() == condition).float().mean()
        )
        stage_acc = ((torch.sigmoid(stage_pred) > 0.5).int() == stage).float().mean()
        subject_acc = (
            ((torch.sigmoid(subject_pred) > 0.5).int() == subject).float().mean()
        )

        self.log(f"{running_stage}_condition_acc", condition_acc)
        self.log(f"{running_stage}_stage_acc", stage_acc)
        self.log(f"{running_stage}_subject_acc", subject_acc)

        return condition_loss + 0 * stage_loss + 0 * subject_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
