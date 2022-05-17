from os.path import join
import numpy as np
import pandas as pd
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci


result_dir = "transformer/data/"
target_type = "baseline-eyes"  # baseline-eyes, fist-motion


def extract_baseline_eyes(subjects, runs, epoch_duration):
    epochs = []
    subject_labels = []
    labels = []
    run2label = {1: "eyes open", 2: "eyes open"}
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
                labels.append(run2label)
                offset += epoch_steps
    return epochs, subject_labels, labels


def extract_fist_motion(subjects, runs, epoch_duration):
    epochs = []
    subject_labels = []
    labels = []
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)

        curr_epochs = mne.Epochs(
            raw, *raw.find_edf_events(), tmin=-1, tmax=epoch_duration - 1, preload=True
        )
        if curr_epochs.get_data().shape[-1] != (epoch_duration * 160 + 1):
            # skip runs which don't have a sampling frequency of 160
            continue
        # left fist
        epochs.extend(list(curr_epochs["T1"].get_data()))
        subject_labels.extend([subject] * len(curr_epochs["T1"]))
        labels.extend(["left fist"] * len(curr_epochs["T1"]))
        # right fist
        epochs.extend(list(curr_epochs["T2"].get_data()))
        subject_labels.extend([subject] * len(curr_epochs["T2"]))
        labels.extend(["right fist"] * len(curr_epochs["T2"]))
    return epochs, subject_labels, labels


def extract_epochs(target, subjects=range(1, 110)):
    if target == "baseline-eyes":
        return extract_baseline_eyes(subjects=subjects, runs=[1, 2], epoch_duration=5)
    elif target == "fist-motion":
        return extract_fist_motion(subjects=subjects, runs=[3, 7, 11], epoch_duration=5)
    raise ValueError(f"Unrecognized target {targer}")


epochs, subject_labels, labels = extract_epochs(target_type)

shape = len(epochs), epochs[0].shape[1], epochs[0].shape[0]
fname = f"nsamp_{shape[0]}-eplen_{shape[1]}-example_{target_type}"
print("\nSaving raw data...", end="")
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
    meta_info.iloc[i] = [subject_labels[i], -1, labels[i]]
print("done")

print("Saving metadata...", end="")
meta_info.to_csv(join(result_dir, "label-" + fname + ".csv"))
print("done")
