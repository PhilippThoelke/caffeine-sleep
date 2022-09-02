from tqdm import tqdm
import glob
from os.path import join
import numpy as np
from mne.filter import filter_data
from scipy.signal import welch
from fooof import FOOOF
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pickle


DATA_PATH = "data/raw_eeg200/"
RESULTS_PATH = "results/final/fooof200/"


def fooof_single_channel(freq, psd, freq_range):
    fm = FOOOF(aperiodic_mode="knee")
    fm.fit(freq, psd, freq_range)
    return fm


def fit_fooof(stage, condition, sfreq=256, freq_range=(0.5, 50), channelwise=False):
    # load data
    paths = glob.glob(join(DATA_PATH, f"*{stage}*{condition}.npy"))
    data = np.concatenate([np.load(path) for path in paths], axis=0)
    # filter data
    data = filter_data(
        data.transpose(0, 2, 1).astype(float), sfreq, 0.5, 50, n_jobs=-1
    ).transpose(0, 2, 1)
    # compute power spectrum
    freq, psd = welch(data, sfreq, nperseg=4 * sfreq, axis=1)

    if channelwise:
        psd = psd.mean(axis=0)
        # fit FOOOF models
        models = Parallel(n_jobs=-1)(
            delayed(fooof_single_channel)(freq, psd[:, ch], freq_range)
            for ch in tqdm(
                range(psd.shape[1]), desc=f"fitting models on {stage}-{condition}"
            )
        )
        return models
    else:
        psd = psd.mean(axis=(0, 2))
        fm = FOOOF()
        fm.fit(freq, psd, freq_range)
        return fm


if __name__ == "__main__":
    sfreq = 256
    fm_caf_all = fit_fooof("NREM", "CAF", sfreq=sfreq, channelwise=True)
    fm_plac_all = fit_fooof("NREM", "PLAC", sfreq=sfreq, channelwise=True)

    with open(join(RESULTS_PATH, "fooof.pkl"), "wb") as f:
        pickle.dump({"CAF": fm_caf_all, "PLAC": fm_plac_all}, f)
