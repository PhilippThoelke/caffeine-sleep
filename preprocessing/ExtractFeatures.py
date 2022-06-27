import os
import glob
import numpy as np
import pandas as pd
from EEGProcessing import (
    load_data,
    extract_sleep_stages,
    load_pre_split_data,
    power_spectral_density,
    shannon_entropy,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    hurst_exponent,
    fooof_1_over_f,
    zero_one_chaos,
    compute_avalanches,
)

# caffeine dose: 200 or 400
CAF_DOSE = 200
# path to the subject information CSV file
SUBJECTS_PATH = f"data/CAF_{CAF_DOSE}_Inventaire.csv"
# directory with the raw EEG data
DATA_PATH = f"data/raw_eeg{CAF_DOSE}"
# directory where features will be stored
FEATURES_PATH = f"data/Features{CAF_DOSE}_redo"
# if True, use a hypnogram to split the raw EEG data into sleep stages
# if False, load data that is already split into sleep stages
SPLIT_STAGES = False

# which features to compute
psd = False
shanEn = False
permEn = False
sampEn = False
specShanEn = False
specPermEn = False
specSampEn = False
hurstExp = False
oneOverF = False
zeroOneChaos = False
avalanche = True


def save_feature_dict(name, folder_path, feature_dict):
    folder_path = os.path.join(folder_path, name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for key, feature in feature_dict.items():
        path = os.path.join(folder_path, f"{name}_{key}")
        np.save(path, feature)


def create_folder(name, folder_path):
    folder_path = os.path.join(folder_path, name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


subject_ids = pd.read_csv(SUBJECTS_PATH, index_col=0)["Subject_id"]

done_subjects = []
while len(subject_ids) > len(done_subjects):
    psd_done = False
    shanEn_done, permEn_done, sampEn_done = [False] * 3
    specShanEn_done, specPermEn_done, specSampEn_done = [False] * 3
    hurstExp_done, oneOverF_done, zeroOneChaos_done, avalanche_done = (
        False,
        False,
        False,
        False,
    )

    subject_id = subject_ids.iloc[0]
    i = 1
    while subject_id in done_subjects:
        if i >= len(subject_ids):
            break
        subject_id = subject_ids.iloc[i]
        i += 1

    subject_path = os.path.join(FEATURES_PATH, subject_id)
    print(
        f"----------------------------- NEW SUBJECT: {subject_id} -----------------------------"
    )
    done_subjects.append(subject_id)
    if not os.path.exists(subject_path):
        print(f"Creating feature folder for subject '{subject_id}'...", end="")
        os.mkdir(subject_path)
        if psd:
            create_folder("PSD", subject_path)
        if shanEn:
            create_folder("ShanEn", subject_path)
        if permEn:
            create_folder("PermEn", subject_path)
        if sampEn:
            create_folder("SampEn", subject_path)
        if specShanEn:
            create_folder("SpecShanEn", subject_path)
        if specPermEn:
            create_folder("SpecPermEn", subject_path)
        if specSampEn:
            create_folder("SpecSampEn", subject_path)
        if hurstExp:
            create_folder("HurstExp", subject_path)
        if oneOverF:
            create_folder("OneOverF", subject_path)
        if zeroOneChaos:
            create_folder("ZeroOneChaos", subject_path)
        if avalanche:
            create_folder("Avalanche", subject_path)
        print("done")
    else:
        features = [
            file.split(os.sep)[-1]
            for file in glob.glob(os.path.join(subject_path, "*"))
        ]
        finished = True
        if psd:
            if "PSD" in features:
                psd_done = True
            else:
                finished = False
                create_folder("PSD", subject_path)
        if shanEn:
            if "ShanEn" in features:
                shanEn_done = True
            else:
                finished = False
                create_folder("ShanEn", subject_path)
        if permEn:
            if "PermEn" in features:
                permEn_done = True
            else:
                finished = False
                create_folder("PermEn", subject_path)
        if sampEn:
            if "SampEn" in features:
                sampEn_done = True
            else:
                finished = False
                create_folder("SampEn", subject_path)
        if specShanEn:
            if "SpecShanEn" in features:
                specShanEn_done = True
            else:
                finished = False
                create_folder("SpecShanEn", subject_path)
        if specPermEn:
            if "SpecPermEn" in features:
                specPermEn_done = True
            else:
                finished = False
                create_folder("SpecPermEn", subject_path)
        if specSampEn:
            if "SpecSampEn" in features:
                specSampEn_done = True
            else:
                finished = False
                create_folder("SpecSampEn", subject_path)
        if hurstExp:
            if "HurstExp" in features:
                hurstExp_done = True
            else:
                finished = False
                create_folder("HurstExp", subject_path)
        if oneOverF:
            if "OneOverF" in features:
                oneOverF_done = True
            else:
                finished = False
                create_folder("OneOverF", subject_path)
        if zeroOneChaos:
            if "ZeroOneChaos" in features:
                zeroOneChaos_done = True
            else:
                finished = False
                create_folder("ZeroOneChaos", subject_path)
        if avalanche:
            if "Avalanche" in features:
                avalanche_done = True
            else:
                finished = False
                create_folder("Avalanche", subject_path)

        if finished:
            print("Features already computed, moving on.")
            continue

    if SPLIT_STAGES:
        eeg_path = os.path.join(DATA_PATH, subject_id, "EEG_data_clean.npy")
        assert os.path.exists(eeg_path), (
            f"Raw EEG data was not found at {eeg_path}. "
            "Make sure it exists or switch SPLIT_STAGES to False."
        )
        hyp_path = os.path.join(DATA_PATH, subject_id, "hyp_clean.npy")
        assert os.path.exists(hyp_path), (
            f"Hypnogram data was not found at {hyp_path}. "
            "Make sure it exists or switch SPLIT_STAGES to False."
        )
        print("Loading EEG and sleep hypnogram data...", end="", flush=True)
        eeg_data, hyp_data = load_data(eeg_path, hyp_path)
        print("done")

        print("Extracting sleep stages...", end="", flush=True)
        stages = extract_sleep_stages(eeg_data, hyp_data)
        del eeg_data, hyp_data
        print("done")
    else:
        print(
            "SPLIT_STAGES is set to False. Loading data that is already split into sleep stages...",
            end="",
            flush=True,
        )
        stages = load_pre_split_data(DATA_PATH, subject_id)
        print("done")

    if psd and not psd_done:
        feature = {}
        print("Computing power spectral density...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = power_spectral_density(stage)
        save_feature_dict("PSD", subject_path, feature)
        print("done")

    if shanEn and not shanEn_done:
        feature = {}
        print("Computing absolute value shannon entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = np.empty((stage.shape[0], stage.shape[2]))
            for elec in range(stage.shape[0]):
                for epoch in range(stage.shape[2]):
                    feature[key][elec, epoch] = shannon_entropy(
                        np.abs(stage[elec, :, epoch])
                    )
        save_feature_dict("ShanEn", subject_path, feature)
        print("done")

    if permEn and not permEn_done:
        feature = {}
        print("Computing permutation entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = np.empty((stage.shape[0], stage.shape[2]))
            for elec in range(stage.shape[0]):
                for epoch in range(stage.shape[2]):
                    feature[key][elec, epoch] = permutation_entropy(
                        stage[elec, :, epoch]
                    )
        save_feature_dict("PermEn", subject_path, feature)
        print("done")

    if sampEn and not sampEn_done:
        feature = {}
        print("Computing sample entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = np.empty((stage.shape[0], stage.shape[2]))
            for elec in range(stage.shape[0]):
                for epoch in range(stage.shape[2]):
                    feature[key][elec, epoch] = sample_entropy(stage[elec, :, epoch])
        save_feature_dict("SampEn", subject_path, feature)
        print("done")

    if specShanEn and not specShanEn_done:
        feature = {}
        print("Computing spectral shannon entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = spectral_entropy(stage, method="shannon")
        save_feature_dict("SpecShanEn", subject_path, feature)
        print("done")

    if specPermEn and not specPermEn_done:
        feature = {}
        print("Computing spectral permutation entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = spectral_entropy(stage, method="permutation")
        save_feature_dict("SpecPermEn", subject_path, feature)
        print("done")

    if specSampEn and not specSampEn_done:
        feature = {}
        print("Computing spectral sample entropy...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = spectral_entropy(stage, method="sample")
        save_feature_dict("SpecSampEn", subject_path, feature)
        print("done")

    if hurstExp and not hurstExp_done:
        feature = {}
        print("Computing hurst exponent...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = hurst_exponent(stage)
        save_feature_dict("HurstExp", subject_path, feature)
        print("done")

    if oneOverF and not oneOverF_done:
        feature = {}
        print("Computing 1/f...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = fooof_1_over_f(stage)
        save_feature_dict("OneOverF", subject_path, feature)
        print("done")

    if zeroOneChaos and not zeroOneChaos_done:
        feature = {}
        print("Computing 0-1 chaos test...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = zero_one_chaos(stage)
        save_feature_dict("ZeroOneChaos", subject_path, feature)
        print("done")

    if avalanche and not avalanche_done:
        feature = {}
        print("Computing avalanches...", end="", flush=True)
        for key, stage in stages.items():
            feature[key] = compute_avalanches(stage)
        save_feature_dict("Avalanche", subject_path, feature)
        print("done")
