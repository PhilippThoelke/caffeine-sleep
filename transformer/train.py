import argparse
from functools import reduce
from os import makedirs, path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from dataset import RawDataset
from module import TransformerModule


def split_data(data, val_subject_ratio):
    unique_subject_ids = np.unique(data.subject_ids)
    num_val_subjects = int(len(unique_subject_ids) * val_subject_ratio)
    subject_idxs = np.random.choice(unique_subject_ids, num_val_subjects, replace=False)
    val_mask = reduce(np.bitwise_or, [data.subject_ids == i for i in subject_idxs])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def main(args):
    # load data
    data = RawDataset(args.data_path, args.label_path)
    idx_train, idx_val = split_data(data, args.val_subject_ratio)

    # train subset
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # val subset
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
    splits = dict(
        train_idx=idx_train,
        val_idx=idx_val,
        train_subjects=data.id2subject(np.unique(data.subject_ids[idx_train])),
        val_subjects=data.id2subject(np.unique(data.subject_ids[idx_val])),
    )
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
        "--val-subject-ratio",
        default=0.2,
        type=float,
        help="ratio of subjects to be used for validation",
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
        "--weight-decay",
        default=0.01,
        type=float,
        help="weight decay",
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
