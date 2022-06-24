import Loader
import re
import os
import pickle
import numpy as np
import pandas as pd


CAF_DOSE = 200
MIN_AGE = -1  # -1 for no minimum age
MAX_AGE = -1  # -1 for no maximum age

FEATURES_PATH = f"data/Features{CAF_DOSE}"
RESULT_PATH = f"data/Features{CAF_DOSE}/Combined"
SUBJECTS_PATH = f"data/CAF_{CAF_DOSE}_Inventaire.csv"
DATA_PATH = "data/"

BANDS = ["delta", "theta", "alpha", "sigma", "beta", "low gamma"]


def get_psd_labels_groups(data_dict):
    print("PSD...")

    # get the labels, load the PSD feature and load the hypnograms
    subject_labels = Loader.load_labels(CAF_DOSE, SUBJECTS_PATH)
    psd = Loader.load_feature("PSD", CAF_DOSE, FEATURES_PATH)

    meta_info = pd.read_csv(SUBJECTS_PATH, index_col=0)

    with open(
        os.path.join(DATA_PATH, f"sample_difference{CAF_DOSE}.pickle"), "rb"
    ) as file:
        drop_counts = pickle.load(file)

    labels = {}
    groups = {}
    group_indices = {}
    group_names = {}

    for stage in psd.keys():
        curr_data = []
        curr_awsl = []
        labels[stage] = []
        groups[stage] = []
        group_indices[stage] = {}
        group_names[stage] = {}

        if stage == "AWA":
            labels["AWSL"] = []
            groups["AWSL"] = []
            group_indices["AWSL"] = {}
            group_names["AWSL"] = {}

        for subject_id, subject in psd[stage].items():
            age = meta_info[meta_info["Subject_id"] == subject_id]["Age"].values[0]
            if MIN_AGE >= 0:
                if age < MIN_AGE:
                    # dropping subject, too young
                    continue
            if MAX_AGE >= 0:
                if age > MAX_AGE:
                    # dropping subject, too old
                    continue

            if subject.size == 0:
                # drop empty subjects
                print(
                    f"Dropping recording {subject_id}, "
                    "missing values for this feature"
                )
                continue

            # add the current subject's data to the list and append its label
            curr_data.append(subject)
            labels[stage] += [subject_labels[subject_id]] * subject.shape[1]

            # manage AWSL stage
            if stage == "AWA":
                curr = subject[:, drop_counts[subject_id] :]
                if curr.size > 0:
                    curr_awsl.append(curr)
                    labels["AWSL"] += [subject_labels[subject_id]] * curr_awsl[
                        -1
                    ].shape[1]

            subject_short = re.match("\S\d+", subject_id)[0]
            if len(group_indices[stage]) == 0:
                # first group gets index 0
                group_indices[stage][subject_short] = 0
            elif not subject_short in group_indices[stage]:
                # not the first group, increase max index by 1
                group_indices[stage][subject_short] = (
                    np.max(list(group_indices[stage].values())) + 1
                )

            # manage AWSL stage
            if stage == "AWA" and subject[:, drop_counts[subject_id] :].size > 0:
                if len(group_indices["AWSL"]) == 0:
                    group_indices["AWSL"][subject_short] = 0
                elif not subject_short in group_indices["AWSL"]:
                    group_indices["AWSL"][subject_short] = (
                        np.max(list(group_indices["AWSL"].values())) + 1
                    )

            # add current index to group indices
            groups[stage] += [group_indices[stage][subject_short]] * subject.shape[1]
            group_names[stage][group_indices[stage][subject_short]] = subject_short

            # manage AWSL stage
            awsl_curr = subject[:, drop_counts[subject_id] :]
            if stage == "AWA" and awsl_curr.size > 0:
                groups["AWSL"] += [
                    group_indices["AWSL"][subject_short]
                ] * awsl_curr.shape[1]
                group_names["AWSL"][
                    group_indices["AWSL"][subject_short]
                ] = subject_short

        # concatenate data from all subjects
        concatenated = np.concatenate(curr_data, axis=1)
        labels[stage] = np.array(labels[stage])
        groups[stage] = np.array(groups[stage])

        if stage == "AWA":
            concatenated_awsl = np.concatenate(curr_awsl, axis=1)
            labels["AWSL"] = np.array(labels["AWSL"])
            groups["AWSL"] = np.array(groups["AWSL"])

        for i, band in enumerate(BANDS):
            if stage not in data_dict:
                data_dict[stage] = dict()
            # add all power bands to the feature dictionary for the current stage
            data_dict[stage][f"PSD_{band}"] = concatenated[:, :, i].T

            if stage == "AWA":
                if "AWSL" not in data_dict:
                    data_dict["AWSL"] = dict()
                data_dict["AWSL"][f"PSD_{band}"] = concatenated_awsl[:, :, i].T

    if "N1" in data_dict:
        data_dict["NREM"] = dict()
        for band in BANDS:
            ft = f"PSD_{band}"
            # combine N1, N2 and N3 into NREM
            nrem = [data_dict["N1"][ft], data_dict["N2"][ft], data_dict["N3"][ft]]
            # add current power band to the NREM features dictionary
            data_dict["NREM"][ft] = np.concatenate(nrem, axis=0)

        # combine N1, N2 and N3 label and group arrays into NREM sleep stage
        labels["NREM"] = np.concatenate(
            [labels["N1"], labels["N2"], labels["N3"]], axis=0
        )
        groups["NREM"] = np.concatenate(
            [groups["N1"], groups["N2"], groups["N3"]], axis=0
        )

    return labels, groups, group_names


def get_feature(data_dict, feature_name):
    print(feature_name, "...", sep="")

    feature = Loader.load_feature(feature_name, CAF_DOSE, FEATURES_PATH)
    meta_info = pd.read_csv(SUBJECTS_PATH, index_col=0)

    with open(
        os.path.join(DATA_PATH, f"sample_difference{CAF_DOSE}.pickle"), "rb"
    ) as file:
        drop_counts = pickle.load(file)

    for stage in feature.keys():
        curr_data = []
        curr_awsl = []

        for subject_id, subject in feature[stage].items():
            age = meta_info[meta_info["Subject_id"] == subject_id]["Age"].values[0]
            if MIN_AGE >= 0:
                if age < MIN_AGE:
                    # dropping subject, too young
                    continue
            if MAX_AGE >= 0:
                if age > MAX_AGE:
                    # dropping subject, too old
                    continue

            if subject.size == 0:
                # drop empty subjects
                print(
                    f"Dropping recording {subject_id}, "
                    "missing values for this feature"
                )
                continue
            curr_data.append(subject)

            # manage AWSL stage
            if stage == "AWA":
                curr = subject[:, drop_counts[subject_id] :]
                if curr.size > 0:
                    curr_awsl.append(curr)
                else:
                    print(f"No AWSL data for subject {subject_id} in {feature_name}")

        data_dict[stage][feature_name] = np.concatenate(curr_data, axis=1).T

        # manage AWSL stage
        if stage == "AWA":
            data_dict["AWSL"][feature_name] = np.concatenate(curr_awsl, axis=1).T

    if "N1" in data_dict and not feature_name in data_dict["NREM"]:
        nrem = [
            data_dict["N1"][feature_name],
            data_dict["N2"][feature_name],
            data_dict["N3"][feature_name],
        ]
        data_dict["NREM"][feature_name] = np.concatenate(nrem, axis=0)


def normalize(data_dict, groups_dict):
    # average data stage- and feature-wise
    for stage in data_dict.keys():
        for feature in data_dict[stage].keys():
            for group in np.unique(groups_dict[stage]):
                mask = groups_dict[stage] == group
                curr = data_dict[stage][feature][mask]
                data_dict[stage][feature][mask] = (curr - curr.mean()) / curr.std()


def normalize_avg(data_avg, groups_avg, data, groups):
    # average data stage- and feature-wise
    for stage in data_avg.keys():
        print(stage)
        for feature in data_avg[stage].keys():
            for group in np.unique(groups_avg[stage]):
                mask_avg = groups_avg[stage] == group
                mask = groups[stage] == group

                curr_avg = data_avg[stage][feature][mask_avg]
                curr = data[stage][feature][mask]
                data_avg[stage][feature][mask_avg] = (
                    curr_avg - curr.mean()
                ) / curr.std()


if __name__ == "__main__":
    data = dict()

    print("-------------------- Concatenating features --------------------")
    labels, groups, names = get_psd_labels_groups(data)
    get_feature(data, "SpecShanEn")
    get_feature(data, "SampEn")
    get_feature(data, "SpecSampEn")
    get_feature(data, "PermEn")
    get_feature(data, "SpecPermEn")
    get_feature(data, "HurstExp")
    get_feature(data, "OneOverF")
    get_feature(data, "ZeroOneChaos")

    print("-------------------- Averaging features --------------------")

    data_avg = {}
    labels_avg = {}
    groups_avg = {}
    for stage, current in data.items():
        data_avg[stage] = {}
        labels_avg[stage] = []
        groups_avg[stage] = []
        print(stage)

        added = set()
        dropped = []
        for feature in current.keys():
            data_avg[stage][feature] = []

            for i in range(len(np.unique(groups[stage]))):
                data_0 = current[feature][(groups[stage] == i) & (labels[stage] == 0)]
                data_1 = current[feature][(groups[stage] == i) & (labels[stage] == 1)]

                if len(data_0) == 0 or len(data_1) == 0:
                    if names[stage][i] not in dropped:
                        dropped.append(names[stage][i])
                        print(
                            f"Dropping subject {names[stage][i]} in stage "
                            f"{stage} ({len(data_0)} plac, {len(data_1)} caf)"
                        )
                    continue

                added.add(i)
                data_avg[stage][feature].append(data_0.mean(axis=0))
                data_avg[stage][feature].append(data_1.mean(axis=0))

            data_avg[stage][feature] = np.array(data_avg[stage][feature])

        labels_avg[stage] += [0, 1] * len(added)
        for i in added:
            groups_avg[stage] += [i, i]

        labels_avg[stage] = np.array(labels_avg[stage])
        groups_avg[stage] = np.array(groups_avg[stage])

    print("-------------------- Normalizing samples --------------------")
    normalize_avg(data_avg, groups_avg, data, groups)
    normalize(data, groups)

    print("-------------------- Saving data --------------------")

    age_suffix = ""
    if MIN_AGE >= 0:
        age_suffix += f"_age_f{MIN_AGE}"
    if MAX_AGE >= 0:
        if MIN_AGE < 0:
            age_suffix += f"_age_t{MAX_AGE}"
        else:
            age_suffix += f"-t{MAX_AGE}"

    with open(os.path.join(RESULT_PATH, f"data{age_suffix}.pickle"), "wb") as file:
        pickle.dump(data, file)

    with open(os.path.join(RESULT_PATH, f"labels{age_suffix}.pickle"), "wb") as file:
        pickle.dump(labels, file)

    with open(os.path.join(RESULT_PATH, f"groups{age_suffix}.pickle"), "wb") as file:
        pickle.dump(groups, file)

    with open(os.path.join(RESULT_PATH, f"data_avg{age_suffix}.pickle"), "wb") as file:
        pickle.dump(data_avg, file)

    with open(
        os.path.join(RESULT_PATH, f"labels_avg{age_suffix}.pickle"), "wb"
    ) as file:
        pickle.dump(labels_avg, file)

    with open(
        os.path.join(RESULT_PATH, f"groups_avg{age_suffix}.pickle"), "wb"
    ) as file:
        pickle.dump(groups_avg, file)

    print("Done!")

