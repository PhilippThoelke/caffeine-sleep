import os
from os.path import join
import glob
import pickle
import numpy as np
from antropy import lziv_complexity
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from mne import viz, stats


data_dir = "../transformer/data/200/"
result_dir = "../results/lziv"

stages = ["AWA", "NREM", "REM"]
conditions = ["CAF", "PLAC"]


def get_paths(root, subject="*", stage="*", condition="*"):
    return glob.glob(join(root, f"{subject}n*_{stage}_*_{condition}.npy"))


def complexity(epoch):
    return lziv_complexity((epoch > np.median(epoch)).astype(int), normalize=True)


result = {}
for stage in stages:
    result[stage] = {}
    for condition in conditions:
        print(f"processing {stage} {condition}:")
        # load data
        paths = get_paths(data_dir, stage=stage, condition=condition)

        data = {}
        for path in tqdm(paths, desc="loading data"):
            subject = path.split(os.sep)[-1].split("_")[0].split("n")[0]
            if subject in data:
                data[subject] = np.concatenate([data[subject], np.load(path)])
            else:
                data[subject] = np.load(path)

        result[stage][condition] = []
        for subject in tqdm(data.keys(), desc="estimating complexity"):
            dat = data[subject].transpose((0, 2, 1)).reshape(-1, 5120)
            res = Parallel(n_jobs=-1)(delayed(complexity)(epoch) for epoch in dat)
            result[stage][condition].append(np.array(res).reshape(-1, 20).mean(axis=0))
        result[stage][condition] = np.stack(result[stage][condition])

    result[stage] = stats.permutation_t_test(
        result[stage]["CAF"] - result[stage]["PLAC"],
        n_permutations=1000,
        tail=0,
        n_jobs=-1,
    )


with open(join(result_dir, "lziv.pkl"), "wb") as f:
    pickle.dump(result, f)
