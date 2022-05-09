from os.path import join
import argparse
import glob
import yaml
from tqdm import tqdm
from functools import partialmethod
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader, Subset
from mne.viz import plot_topomap
from mne.stats import permutation_t_test
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


def main(
    model_path,
    data_path,
    label_path,
    result_dir,
    sensors_path,
    sleep_stage="all",
    data_splits_path=None,
    rollout=True,
    n_batches=-1,
):
    torch.set_grad_enabled(False)

    # load model
    model = TransformerModule.load_from_checkpoint(model_path, num_subjects=40).eval()
    model.freeze()
    model = attention_wrapper(model)

    # load data
    data = RawDataset(data_path, label_path, stage=sleep_stage)
    if data_splits_path is not None:
        # only use validation data
        val_idx = torch.load(data_splits_path)["val_idx"]
        data = Subset(data, val_idx)
    dl = DataLoader(data, batch_size=32, num_workers=4, shuffle=True)

    # extract attention weights
    weights, labels = [], []
    accs = []
    for x, y, _, _ in tqdm(dl, desc="extracting attention weights"):
        pred, w = model(x)

        mask = (pred.squeeze() < 0.5).int() == y
        w = w[:, mask]
        y = y[mask]

        accs.append(mask.float().mean().item())

        if rollout:
            w = attention_rollout(w)

        weights.append(w)
        labels.append(y)

        if len(weights) == n_batches:
            break

    print(f"accuracy: {np.mean(accs):.2f}")

    labels = torch.cat(labels)

    pos = loadmat(sensors_path)["Cor"].T
    pos = np.array([pos[1], pos[0]]).T

    if rollout:
        # plot weights with rollout
        weights = torch.cat(weights)
        weights1 = weights[labels == 0].mean(dim=0)
        weights2 = weights[labels == 1].mean(dim=0)

        fig, (ax1, ax2) = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(8, 5)
        )
        fig.suptitle("attention weight rollout", fontsize=15)
        ax1.imshow(weights1)
        ax1.set_title(data.id2condition(0))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_ylabel("tokens")
        ax1.set_xlabel("tokens")
        ax2.imshow(weights2)
        ax2.set_title(data.id2condition(1))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("tokens")
        plt.tight_layout()
        plt.savefig(join(result_dir, "rollout.png"), dpi=300)

        # topoplots
        elecs1 = weights1.mean(dim=0)[1:].reshape(20, model.model.hparams.num_tokens)
        elecs2 = weights2.mean(dim=0)[1:].reshape(20, model.model.hparams.num_tokens)

        vmin = min(elecs1.min(), elecs2.min())
        vmax = max(elecs1.max(), elecs2.max())

        # channels and temporal
        fig, axes = plt.subplots(nrows=2, ncols=20, figsize=(12, 3))
        fig.suptitle("mean attention weights across time and channels", fontsize=15)
        for i, (elec, ax) in enumerate(zip(elecs1.T, axes[0])):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
            ax.set_title(f"t={i}")
        for elec, ax in zip(elecs2.T, axes[1]):
            plot_topomap(
                elec, pos, axes=ax, show=False, contours=False, vmin=vmin, vmax=vmax
            )
        axes[0, 0].set_ylabel(data.id2condition(0))
        axes[1, 0].set_ylabel(data.id2condition(1))
        plt.tight_layout(pad=0.1, h_pad=0)
        plt.savefig(join(result_dir, "temporal.png"), dpi=300)

        # averaged temporally
        fig, axes = plt.subplots(ncols=2)
        plot_topomap(
            elecs1.mean(dim=1),
            pos,
            axes=axes[0],
            show=False,
            contours=False,
            vmin=vmin,
            vmax=vmax,
        )
        plot_topomap(
            elecs2.mean(dim=1),
            pos,
            axes=axes[1],
            show=False,
            contours=False,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(data.id2condition(0))
        axes[1].set_title(data.id2condition(1))
        plt.savefig(join(result_dir, "temporally-averaged.png"), dpi=300)

        # averaged temporally, caf - plac
        temporal_agg_fn = "mean"
        caf_id = int(data.id2condition(0) == "PLAC")
        w = weights.mean(dim=1)[:, 1:].reshape(-1, 20, model.model.hparams.num_tokens)
        if temporal_agg_fn == "max":
            w = w.max(dim=2).values
        elif temporal_agg_fn == "mean":
            w = w.mean(dim=2)
        else:
            raise RuntimeError(f"unknown {temporal_agg_fn}")
        caf = w[labels == caf_id]
        plac = w[labels == (1 - caf_id)]
        mincount = min(len(caf), len(plac))
        caf = caf[:mincount].numpy()
        plac = plac[:mincount].numpy()
        tval, pval, _ = permutation_t_test(caf - plac, n_permutations=10000, n_jobs=-1)

        p_thresh = 0.01

        plt.figure()
        plt.title(
            (
                f"t-scores of (CAF - PLAC)\n"
                f"aggregated temporally using {temporal_agg_fn}\n"
                f"significance at p<{p_thresh}"
            )
        )

        absmax = max(abs(tval.min()), abs(tval.max()))
        plot_topomap(
            tval,
            pos,
            mask=pval < p_thresh,
            show=False,
            contours=False,
            vmin=-absmax,
            vmax=absmax,
            cmap="coolwarm",
        )
        plt.tight_layout()
        plt.savefig(join(result_dir, "t-test.png"), dpi=300)
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
        "--model-dir",
        type=str,
        required=True,
        help="path to the pytorch-lightning experiment directory",
    )
    parser.add_argument(
        "--sensors-path",
        type=str,
        required=True,
        help="path to the sensor locations file",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="path to the directory where the figures should be stored",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="manually specify where the data file is (optional)",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default=None,
        help="manually specify where the label file is (optional)",
    )
    args = parser.parse_args()

    with open(join(args.model_dir, "hparams.yaml"), "r") as f:
        hparams = yaml.full_load(f)

    model_path = glob.glob(join(args.model_dir, "checkpoints", "*.ckpt"))[0]

    # run the analysis
    main(
        model_path,
        hparams["data_path"] if args.data_path is None else args.data_path,
        hparams["label_path"] if args.label_path is None else args.label_path,
        args.result_dir,
        args.sensors_path,
        hparams["sleep_stage"],
        join(args.model_dir, "splits.pt"),
    )
