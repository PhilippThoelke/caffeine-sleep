import os
from os.path import join
import glob
import pickle
import numpy as np
from scipy import signal
from tqdm import tqdm


data_dir = "transformer/data/200/"
result_dir = "results/psd/"

stages = ["AWSL", "NREM", "REM"]
conditions = ["CAF", "PLAC"]


def get_paths(root, subject="*", stage="*", condition="*"):
    return glob.glob(join(root, f"{subject}n*_{stage}_*_{condition}.npy"))


def compute_psd(epoch):
    return signal.welch(
        epoch, fs=256, nperseg=len(epoch) // 6, noverlap=0, window="hamming"
    )


result = {}
for stage in stages:
    result[stage] = {}
    for condition in conditions:
        result[stage][condition] = {}
        paths = get_paths(data_dir, stage=stage, condition=condition)
        for path in tqdm(paths, desc=f"processing {stage} {condition}"):
            subject = path.split(os.sep)[-1].split("_")[0].split("n")[0]
            data = np.load(path)
            for epoch in data:
                curr_freqs = None
                curr_psds = None
                for elec in range(data.shape[-1]):
                    freq, psd = compute_psd(epoch[..., elec])
                    if curr_freqs is None:
                        curr_freqs = freq[None]
                    else:
                        curr_freqs = np.concatenate([curr_freqs, freq[None]])
                    if curr_psds is None:
                        curr_psds = psd[None]
                    else:
                        curr_psds = np.concatenate([curr_psds, psd[None]])

                if subject in result[stage][condition]:
                    result[stage][condition][subject] = (
                        np.concatenate(
                            [
                                result[stage][condition][subject][0],
                                curr_freqs.T[None],
                            ]
                        ),
                        np.concatenate(
                            [
                                result[stage][condition][subject][1],
                                curr_psds.T[None],
                            ]
                        ),
                    )
                else:
                    result[stage][condition][subject] = (
                        curr_freqs.T[None],
                        curr_psds.T[None],
                    )


with open(join(result_dir, f"psd.pkl"), "wb") as f:
    pickle.dump(result, f)
