import argparse
from tqdm import tqdm
from functools import partialmethod
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader
from mne.viz import plot_topomap
from dataset import RawDataset
from module import TransformerModule


def attention_rollout(weights, normalize_weights=True):
    result = torch.eye(weights.size(-1))
    for weight in weights:
        if normalize_weights:
            weight = weight / weight.sum(dim=-1, keepdims=True)
        result = torch.matmul(weight, result)
    return result


def modified_sa_block(self, x, attn_mask, key_padding_mask, attn_weights):
    x, weights = self.self_attn(
        x,
        x,
        x,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    attn_weights.append(weights)
    return self.dropout1(x)


class attention_wrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        weights = []
        TransformerEncoderLayer._sa_block = partialmethod(
            modified_sa_block, attn_weights=weights
        )
        y = self.model(x)
        return y, torch.stack(weights)


def main(args, rollout=True):
    torch.set_grad_enabled(False)

    # load model
    model = TransformerModule.load_from_checkpoint(args.model_path).eval()
    model.freeze()
    model = attention_wrapper(model)

    # load data
    data = RawDataset(args.data_path, args.label_path)
    dl = DataLoader(data, batch_size=32, num_workers=4, shuffle=True)

    # extract attention weights
    weights, labels = [], []
    for x, y, _, _ in tqdm(dl, desc="extracting attention weights"):
        pred, w = model(x)

        mask = (torch.sigmoid(pred.squeeze()) < 0.5).int() == y
        w = w[:, mask]
        y = y[mask]

        if rollout:
            w = attention_rollout(w)

        weights.append(w)
        labels.append(y)
        if len(weights) > 5:
            break
    labels = torch.cat(labels)

    pos = loadmat("/home/philipp/Documents/caffeine-sleep/data/Coo_caf.mat")["Cor"].T
    pos = np.array([pos[1], pos[0]]).T

    if rollout:
        # plot weights with rollout
        weights = torch.cat(weights)
        weights1 = weights[labels == 0].mean(dim=0)
        weights2 = weights[labels == 1].mean(dim=0)

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(weights1)
        ax1.set_title(data.id2condition(0))
        ax2.imshow(weights2)
        ax2.set_title(data.id2condition(1))

        # topoplots
        elecs1 = weights1.mean(dim=0)[1:].reshape(20, model.model.hparams.num_tokens)
        elecs2 = weights2.mean(dim=0)[1:].reshape(20, model.model.hparams.num_tokens)

        vmin = min(elecs1.min(), elecs2.min())
        vmax = max(elecs1.max(), elecs2.max())

        fig, axes = plt.subplots(nrows=2, ncols=20)
        for elec, ax in zip(elecs1, axes[0]):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
        for elec, ax in zip(elecs2, axes[1]):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
        axes[0, 0].set_ylabel(data.id2condition(0))
        axes[1, 0].set_ylabel(data.id2condition(1))
        plt.show()
    else:
        # plot weights without rollout
        weights = torch.cat(weights, dim=1)
        weights1 = weights[:, labels == 0].mean(dim=1)
        weights2 = weights[:, labels == 1].mean(dim=1)

        fig, axes = plt.subplots(2, len(weights))
        for i, (ax, w) in enumerate(zip(axes[0], weights1)):
            ax.imshow(w)
            ax.set_title(f"layer {i}")
        for ax, w in zip(axes[1], weights2):
            ax.imshow(w)
        axes[0, 0].set_ylabel(data.id2condition(0))
        axes[1, 0].set_ylabel(data.id2condition(1))

        # topoplots
        elecs1 = (
            weights1.mean(dim=1)[:, 1:]
            .reshape(-1, 20, model.model.hparams.num_tokens)
            .mean(dim=-1)
        )
        elecs2 = (
            weights2.mean(dim=1)[:, 1:]
            .reshape(-1, 20, model.model.hparams.num_tokens)
            .mean(dim=-1)
        )

        vmin = min(elecs1.min(), elecs2.min())
        vmax = max(elecs1.max(), elecs2.max())

        fig, axes = plt.subplots(nrows=2, ncols=4)
        for i, (elec, ax) in enumerate(zip(elecs1, axes[0])):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
            ax.set_title(f"layer {i}")
        for elec, ax in zip(elecs2, axes[1]):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
        axes[0, 0].set_ylabel(data.id2condition(0))
        axes[1, 0].set_ylabel(data.id2condition(1))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
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

    args = parser.parse_args()
    main(args)
