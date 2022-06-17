import os
import glob
import numpy as np
import pandas as pd

STAGES = ["AWA", "N1", "N2", "N3", "REM"]


def load_feature(feature_name, caf_dose, features_path):
    """
    Loads one feature into a dictionary (sleep stages) of dictionaries (subjects).

    Args:
        feature_name: the name of the feature, must be the same as the folder name for this feature
        caf_dose: the caffeine dose of the group for which the feature should be loaded (200 or 400)
        features_path: path to the directory where the features are stored

    Returns:
        dictionary with sleep stages as keys containing dictionaries as values, which have the subject ids as keys and as values the data in a numpy array (array shape varies for individual features)
        structure: {stage1: {subject1: data1, subject2: data2, ...}, ...}
    """
    # gets the paths to the folders where the specified feature is stored
    subject_paths = glob.glob(os.path.join(features_path, "*", feature_name))

    feature = {}
    for path in subject_paths:
        # extract the subject id from the current path (second to last element in the path)
        subject_id = path.split(os.sep)[-2]

        # get all stages for the current subject
        stages = set(
            [
                p.split(os.sep)[-1].split("_")[-1].split(".")[0]
                for p in glob.glob(os.path.join(path, "*.npy"))
            ]
        )
        for stage in stages:
            if stage not in feature:
                feature[stage] = {}
            # load the file containing the data for the current stage and subject
            feature[stage][subject_id] = np.load(
                os.path.join(path, f"{feature_name}_{stage}.npy")
            )
    return feature


def load_labels(caf_dose, subjects_path):
    """
    Loads the labels for all subjects into a dictionary.

    Args:
        caf_dose: the caffeine dose of the group for which the labels should be loaded (200 or 400)
        subjects_path: path to the directory where the subject metadata csv file is stored

    Returns:
        dictionary with subject ids as keys and the label as a value (0: caffeine, 1: placebo)
    """
    # read metadata csv file
    subjects = pd.read_csv(subjects_path, index_col=0)[["Subject_id", "CAF"]]

    if caf_dose == 200:
        # for caffeine dose 200 the label names are 'Y' and 'N'
        subjects["CAF"] = (subjects["CAF"] == "Y").values.astype(np.int)
    elif caf_dose == 400:
        # for caffeine dose 400 the label names are 1 and 0
        subjects["CAF"] = (subjects["CAF"] == 1).values.astype(np.int)

    return dict(subjects.values)


def load_hypnograms(caf_dose, subjects_path, data_path):
    """
    Loads hypnograms of all subjects and returns them as a dictionary.

    Args:
        caf_dose: the caffeine dose of the group for which the labels should be loaded (200 or 400)
        subjects_path: path to the directory where the subject metadata csv file is stored
        data_path: path to the directory where the hypnograms are stored

    Returns:
        dictionary with subject ids as keys and the hypnogram as a value (1D numpy array)
    """
    subject_ids = pd.read_csv(subjects_path, index_col=0)["Subject_id"]
    hypnograms = {}
    for subject_id in subject_ids:
        hyp_path = os.path.join(data_path, subject_id, "hyp_clean.npy")
        hypnograms[subject_id] = np.load(hyp_path)
    return hypnograms


def prepare_features(features, subject_labels):
    """
    Combines feature dicts to one numpy array and creates a label vector for the combined features.

    Args:
        features: list of tuples containing the feature dict and column count [(feat1, cols1), (feat2, cols2), ...]
        subject_labels: a dictionary with subject ids as keys and label names as values

    Returns:
        data array containing the combined feature values (sample count x sum(column count for each feature))
        label vector containing label names for each sample
    """
    data = {}
    labels = {}
    for stage in STAGES:
        labels[stage] = []
        features_combined = []

        for subject in subject_labels.keys():
            current = []
            for feature, columns in features:
                if feature[stage][subject].size == 0:
                    # do not look at empty arrays
                    continue
                # collect features for current stage and subject
                if len(feature[stage][subject].shape) == 2:
                    # feature is 2-dimensional, just use transpose
                    current.append(feature[stage][subject].T)
                elif len(feature[stage][subject].shape) == 3:
                    # feature is 3-dimensional, manually reshape to 2-dimensional
                    # np.reshape does not work here
                    reshaped = []
                    for electrode in range(feature[stage][subject].shape[0]):
                        for band in range(feature[stage][subject].shape[2]):
                            if len(feature[stage][subject].shape) != 3:
                                continue
                            reshaped.append(feature[stage][subject][electrode, :, band])
                    current.append(np.array(reshaped).T)

            if len(current) == 0:
                continue

            # merge the features for the current stage and subject
            features_combined.append(np.concatenate(current, axis=1))

            # concatenate the label name for the current subject as often as there are samples
            labels[stage] += [subject_labels[subject]] * features_combined[-1].shape[0]

        # concatenate the features for all subjects
        data[stage] = np.concatenate(features_combined, axis=0)
        labels[stage] = np.array(labels[stage])

    return data, labels
