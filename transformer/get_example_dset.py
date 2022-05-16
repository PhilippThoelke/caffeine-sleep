from os.path import join
import numpy as np
import pandas as pd
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci


subjects = range(1, 110)
runs = [1, 2]
run_labels = {1: "baseline eyes open", 2: "baseline eyes closed"}
epoch_duration = 20
result_dir = "transformer/data/"


epochs = []
subject_labels = []
labels = []
for subject in subjects:
    for run in runs:
        raw_fnames = eegbci.load_data(subject, run)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)
        data = raw.get_data()
        epoch_steps = int(epoch_duration * raw.info["sfreq"])

        offset = 0
        for _ in range(data.shape[1] // epoch_steps):
            epochs.append(data[:, offset : offset + epoch_steps].astype(np.float32))
            subject_labels.append(subject)
            labels.append(run)
            offset += epoch_steps

shape = len(epochs), epochs[0].shape[1], epochs[0].shape[0]
fname = (
    f"nsamp_{shape[0]}-"
    f"eplen_{epoch_duration}-"
    f"runs_{','.join(map(str, runs))}-"
    f"example_dset"
)
file = np.memmap(
    join(result_dir, "raw-" + fname + ".dat"), mode="w+", dtype=np.float32, shape=shape
)
meta_info = pd.DataFrame(
    index=np.arange(shape[0], dtype=int),
    columns=["subject", "stage", "condition"],
    dtype=str,
)

for i in range(shape[0]):
    file[i] = epochs[i].T
    file.flush()
    meta_info.iloc[i] = [subject_labels[i], -1, run_labels[labels[i]]]

meta_info.to_csv(join(result_dir, "label-" + fname + ".csv"))
