import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from os.path import join, basename


def get_paths(root, subject="*", stage="*", condition="*"):
    return glob.glob(join(root, f"{subject}n*_{stage}_*_{condition}.npy"))


def get_total_num_samples(root, **kwargs):
    num_samples = 0
    for path in get_paths(root, **kwargs):
        with open(path, "rb") as f:
            np.lib.format.read_magic(f)
            num_samples += np.lib.format.read_array_header_1_0(f)[0][0]
    return num_samples


def combine_raw_files(dest, root, **kwargs):
    shape = (get_total_num_samples(root, **kwargs), 5120, 20)
    fname = (
        f"nsamp_{shape[0]}-"
        f"subj_{kwargs.get('subject', 'all')}-"
        f"stage_{kwargs.get('stage', 'all')}-"
        f"cond_{kwargs.get('condition', 'all')}"
    )
    file = np.memmap(
        join(dest, "raw-" + fname + ".dat"), mode="w+", dtype=np.float32, shape=shape
    )
    meta_info = pd.DataFrame(
        index=np.arange(shape[0], dtype=int),
        columns=["subject", "stage", "condition"],
        dtype=str,
    )

    paths = get_paths(root, **kwargs)
    curr_idx = 0
    for path in tqdm(paths):
        subj, stage, _, _, cond = basename(path).split(".")[0].split("_")
        subj = subj.split("n")[0]
        data = np.load(path)
        file[curr_idx : curr_idx + data.shape[0]] = data
        meta_info.iloc[curr_idx : curr_idx + data.shape[0]] = [
            [subj, stage, cond]
        ] * data.shape[0]
        curr_idx += data.shape[0]
        file.flush()
    meta_info.to_csv(join(dest, "label-" + fname + ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="directory where the file will be saved",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="path to the raw .npy files"
    )
    parser.add_argument(
        "--subject", type=str, default="*", help="glob string matching the subject IDs"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="*",
        choices=["AWA", "AWSL", "NREM", "REM"],
        help="glob string matching the sleep stage",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="*",
        choices=["CAF", "PLAC"],
        help="glob string matching the condition (caffeine/placebo)",
    )
    args = parser.parse_args()

    combine_raw_files(
        args.destination,
        args.raw_dir,
        subject=args.subject,
        stage=args.stage,
        condition=args.condition,
    )