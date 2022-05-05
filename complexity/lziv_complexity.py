import os
from os.path import join
import glob
import pickle
import numpy as np
from scipy import signal
from antropy import lziv_complexity
from neurokit2.complexity import complexity_lempelziv
from joblib import Parallel, delayed
from tqdm import tqdm
from mne import viz, stats


data_dir = "transformer/data/200/"
result_dir = "results/lziv/"

use_psd = False
use_neurokit = True

stages = ["AWSL", "NREM", "REM"]
conditions = ["CAF", "PLAC"]


def get_paths(root, subject="*", stage="*", condition="*"):
    return glob.glob(join(root, f"{subject}n*_{stage}_*_{condition}.npy"))


def complexity(epoch, use_psd=False, use_neurokit=False):
    if use_psd:
        # TODO: maybe use something other than median split to binarize the PSD
        epoch = signal.welch(
            epoch, fs=256, nperseg=len(epoch) // 6, noverlap=0, window="hamming"
        )[1]
    if use_neurokit:
        return complexity_lempelziv(epoch, normalize=True)[0]
    else:
        return lziv_complexity((epoch > np.median(epoch)).astype(int), normalize=True)


result = {}
for stage in stages:
    result[stage] = {}
    data = {}
    for condition in conditions:
        print(f"processing {stage} {condition}:")
        # load data
        paths = get_paths(data_dir, stage=stage, condition=condition)

        data[condition] = {}
        for path in tqdm(paths, desc="loading data"):
            subject = path.split(os.sep)[-1].split("_")[0].split("n")[0]
            if subject in data[condition]:
                data[condition][subject] = np.concatenate(
                    [data[condition][subject], np.load(path)]
                )
            else:
                data[condition][subject] = np.load(path)

    # drop subjects that don't appear in all conditions
    drop_subjects = set.symmetric_difference(*map(set, data.values()))

    for condition in conditions:
        result[stage][condition] = []
        for subject in tqdm(data[condition].keys(), desc="estimating complexity"):
            if subject in drop_subjects:
                continue
            dat = data[condition][subject].transpose((0, 2, 1)).reshape(-1, 5120)
            res = Parallel(n_jobs=-1)(
                delayed(complexity)(epoch, use_psd, use_neurokit) for epoch in dat
            )
            result[stage][condition].append(np.array(res).reshape(-1, 20).mean(axis=0))
        result[stage][condition] = np.stack(result[stage][condition])

    result[stage] = stats.permutation_t_test(
        result[stage]["CAF"] - result[stage]["PLAC"],
        n_permutations=1000,
        tail=0,
        n_jobs=-1,
    )


with open(join(result_dir, f"lziv{'_psd' if use_psd else ''}.pkl"), "wb") as f:
    pickle.dump(result, f)
