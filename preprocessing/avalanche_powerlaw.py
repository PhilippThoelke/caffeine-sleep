import sys

sys.path.append("caffeine-sleep/preprocessing")

import os
from os.path import join
import glob
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from avalanche import fit_powerlaw, fit_third_exponent


FEATURE_NAME = "Avalanche"
FEATURES_PATH = "data/Features200/"
SUBJECTS_PATH = "data/CAF_200_Inventaire.csv"
STAGES = ["AWSL", "NREM", "REM"]

subj_info = pd.read_csv(SUBJECTS_PATH, index_col=0)


def process_file(path):
    subj = path.split(os.sep)[-3]
    curr = np.load(path, allow_pickle=True)
    cond = (
        "CAF"
        if subj_info[subj_info["Subject_id"] == subj]["CAF"].values[0] == "Y"
        else "PLAC"
    )

    # avalanche size
    sizes = np.array([av["size"] for av in curr])
    sizes_powerlaw = fit_powerlaw(sizes)
    # avalanche duration
    durations = np.array([av["dur_bin"] for av in curr])
    durations_powerlaw = fit_powerlaw(durations)
    # fit third exponent
    third_exp = fit_third_exponent(sizes, durations)
    return cond, sizes_powerlaw, durations_powerlaw, third_exp


# fit powerlaws and compute third exponent
data = {}
for stage in tqdm(STAGES, position=0, desc="sleep stage"):
    data[stage] = {
        "CAF": {"size": [], "dur": [], "third": []},
        "PLAC": {"size": [], "dur": [], "third": []},
    }
    result = Parallel(n_jobs=-1)(
        delayed(process_file)(path)
        for path in tqdm(
            glob.glob(join(FEATURES_PATH, "*", FEATURE_NAME, f"*_{stage}.npy")),
            position=1,
            desc="file",
        )
    )

    for cond, size, dur, third in result:
        data[stage][cond]["size"].append(size)
        data[stage][cond]["dur"].append(dur)
        data[stage][cond]["third"].append(third)

# save third exponent and fitted powerlaws for size and duration
with open("avalanche.pkl", "wb") as f:
    pickle.dump(data, f)
