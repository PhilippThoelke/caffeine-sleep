import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from dataset import RawDataset
from module import TransformerModule


def main(args):
    # load data
    data = RawDataset(args.data_path, args.label_path)
    idxs = torch.randperm(len(data))

    # train subset
    idx_train = idxs[: -int(len(data) * args.val_ratio)]
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # val subset
    idx_val = idxs[-int(len(data) * args.val_ratio) :]
    val_data = Subset(data, idx_val)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, num_workers=4)

    # define model
    module = TransformerModule(args)

    # train
    trainer = pl.Trainer(accelerator="auto", devices="auto")
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
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.2,
        type=float,
        help="ratio of the data to be used for validation",
    )
    parser.add_argument(
        "--num-tokens",
        default=20,
        type=int,
        help="number of temporal tokens the 20s EEG is split into",
    )
    parser.add_argument(
        "--embedding-dim",
        default=256,
        type=int,
        help="dimension of tokens inside the transformer",
    )
    parser.add_argument(
        "--num-layers",
        default=6,
        type=int,
        help="number of encoder layers in the transformer",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout ratio",
    )

    args = parser.parse_args()
    main(args)
