from os import makedirs, path
import argparse
from tqdm import tqdm
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

    # compute data mean and std
    result = [
        (sample[0].mean(), sample[0].std())
        for sample in tqdm(
            DataLoader(train_data, batch_size=256, num_workers=8),
            desc="extracting mean and standard deviation",
        )
    ]
    means, stds = zip(*result)
    mean, std = torch.tensor(means).mean(), torch.tensor(stds).mean()

    # define model
    module = TransformerModule(args, mean, std)

    # define trainer instance
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=args.max_epochs)

    # store train val splits
    makedirs(trainer.log_dir, exist_ok=True)
    splits = dict(train=idx_train, val=idx_val)
    torch.save(splits, path.join(trainer.log_dir, "splits.pt"))

    # train model
    trainer.fit(model=module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to the memory mapped data file",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        required=True,
        help="path to the csv file containing labels",
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-4,
        type=float,
        help="base learning rate",
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
        default=64,
        type=int,
        help="dimension of tokens inside the transformer",
    )
    parser.add_argument(
        "--num-layers",
        default=4,
        type=int,
        help="number of encoder layers in the transformer",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout ratio",
    )
    parser.add_argument(
        "--warmup-steps",
        default=1000,
        type=int,
        help="number of steps for lr warmup",
    )
    parser.add_argument(
        "--max-epochs",
        default=300,
        type=int,
        help="maximum number of epochs",
    )

    args = parser.parse_args()
    main(args)
