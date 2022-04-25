import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from dataset import RawDataset
from module import TransformerModule


def main(data_path, label_path, val_ratio=0.2):
    # load data
    data = RawDataset(data_path, label_path)
    idxs = torch.randperm(len(data))

    # train subset
    idx_train = idxs[: -int(len(data) * val_ratio)]
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

    # val subset
    idx_val = idxs[-int(len(data) * val_ratio) :]
    val_data = Subset(data, idx_val)
    val_dl = DataLoader(val_data, batch_size=64, num_workers=4)

    # define model
    module = TransformerModule()

    # train
    trainer = pl.Trainer()
    trainer.fit(model=module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="path to the memory mapped data file",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        required=True,
        help="path to the csv file containing labels",
    )
    args = parser.parse_args()

    main(args.data_path, args.label_path)
