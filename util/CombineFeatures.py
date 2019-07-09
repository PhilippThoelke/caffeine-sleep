import sys
sys.path.append('..')

from caffeine import Loader
import re
import os
import pickle
import numpy as np
import pandas as pd


CAF_DOSE = 200
MIN_AGE = -1
MAX_AGE = -1

PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\Features{dose}\\Combined'.format(dose=CAF_DOSE)
SUBJECTS_PATH = 'E:\\Cafeine_data\\CAF_{dose}_Inventaire.csv'.format(dose=CAF_DOSE)

STAGES = ['AWA', 'AWSL', 'N1', 'N2', 'N3', 'NREM', 'REM']
BANDS = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'low gamma']


def get_psd_labels_groups(data_dict):
    # get the labels, load the PSD feature and load the hypnograms
    subject_labels = Loader.load_labels(CAF_DOSE)
    psd = Loader.load_feature('PSD', CAF_DOSE)
    hyp = Loader.load_hypnograms(CAF_DOSE)

    meta_info = pd.read_csv(SUBJECTS_PATH, index_col=0)

    labels = {}
    groups = {}
    group_indices = {}
    group_names = {}

    hypnograms = []

    for stage in psd.keys():
        curr_data = []
        labels[stage] = []
        groups[stage] = []
        group_indices[stage] = {}
        group_names[stage] = {}

        for subject_id, subject in psd[stage].items():
            age = meta_info[meta_info['Subject_id'] == subject_id]['Age'].values[0]
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
                continue

            if stage == 'AWA':
                # append current subject's hypnogram
                hypnograms.append(hyp[subject_id])

            # add the current subject's data to the list and append its label
            curr_data.append(subject)
            labels[stage] += [subject_labels[subject_id]] * subject.shape[1]

            subject_short = re.match('\S\d+', subject_id)[0]
            if len(group_indices[stage]) == 0:
                # first group gets index 0
                group_indices[stage][subject_short] = 0
            elif not subject_short in group_indices[stage]:
                # not the first group, increase max index by 1
                group_indices[stage][subject_short] = np.max(
                    list(group_indices[stage].values())) + 1

            # add current index to group indices
            groups[stage] += [group_indices[stage][subject_short]] * subject.shape[1]
            group_names[stage][group_indices[stage][subject_short]] = subject_short

        # concatenate data from all subjects
        concatenated = np.concatenate(curr_data, axis=1)
        labels[stage] = np.array(labels[stage])
        groups[stage] = np.array(groups[stage])

        for i, band in enumerate(BANDS):
            # add all power bands to the feature dictionary for the current stage
            data_dict[stage][f'PSD_{band}'] = concatenated[:, :, i].T

    for band in BANDS:
        ft = f'PSD_{band}'
        # combine N1, N2 and N3 into NREM
        nrem = [data_dict['N1'][ft],
                data_dict['N2'][ft],
                data_dict['N3'][ft]]
        # add current power band to the NREM features dictionary
        data_dict['NREM'][ft] = np.concatenate(nrem, axis=0)

    # combine N1, N2 and N3 label and group arrays into NREM sleep stage
    labels['NREM'] = np.concatenate([labels['N1'], labels['N2'], labels['N3']], axis=0)
    groups['NREM'] = np.concatenate([groups['N1'], groups['N2'], groups['N3']], axis=0)

    return labels, groups, group_names, hypnograms


def get_entropy(data_dict, entropy_type):
    entropy = Loader.load_feature(entropy_type, CAF_DOSE)
    meta_info = pd.read_csv(SUBJECTS_PATH, index_col=0)

    for stage in entropy.keys():
        curr_data = []
        for subject_id, subject in entropy[stage].items():
            age = meta_info[meta_info['Subject_id'] == subject_id]['Age'].values[0]
            if MIN_AGE >= 0:
                if age < MIN_AGE:
                    # dropping subject, too young
                    continue
            if MAX_AGE >= 0:
                if age > MAX_AGE:
                    # dropping subject, too old
                    continue

            if subject.size == 0:
                print(f'Dropping recording {subject_id}')
                continue
            curr_data.append(subject)

        data_dict[stage][entropy_type] = np.concatenate(curr_data, axis=1).T

    nrem = [data_dict['N1'][entropy_type],
            data_dict['N2'][entropy_type],
            data_dict['N3'][entropy_type]]
    data_dict['NREM'][entropy_type] = np.concatenate(nrem, axis=0)


def normalize(data_dict):
    for stage in data_dict.keys():
        for feature in data_dict[stage].keys():
            curr = data_dict[stage][feature]
            data_dict[stage][feature] = (curr - curr.mean(axis=0)) / curr.std(axis=0)


if __name__ == '__main__':
    data = dict([(stage, {}) for stage in STAGES])

    print('-------------------- Concatenating features --------------------')
    labels, groups, names, hypnograms = get_psd_labels_groups(data)
    get_entropy(data, 'SpecShanEn')
    get_entropy(data, 'SpecPermEn')
    get_entropy(data, 'SpecSampEn')
    get_entropy(data, 'SampEn')

    normalize(data)

    # dropping AWA data before first sleep
    for ft in data['AWA'].keys():
        offset = 0
        filtered_data = []
        filtered_groups = []
        filtered_labels = []

        for total, drop_index in [(sum(hyp == 0), np.where(hyp != 0)[0][0]) for hyp in hypnograms]:
            filtered_data.append(data['AWA'][ft][offset+drop_index:offset+total])
            filtered_groups.append(groups['AWA'][offset+drop_index:offset+total])
            filtered_labels.append(labels['AWA'][offset+drop_index:offset+total])
            offset += total

        data['AWSL'][ft] = np.concatenate(filtered_data, axis=0)
        groups['AWSL'] = np.concatenate(filtered_groups, axis=0)
        labels['AWSL'] = np.concatenate(filtered_labels, axis=0)
        names['AWSL'] = names['AWA']

    print('-------------------- Averaging features --------------------')

    data_avg = {}
    labels_avg = {}
    groups_avg = {}
    for stage, current in data.items():
        data_avg[stage] = {}
        labels_avg[stage] = []
        groups_avg[stage] = []
        print(stage)

        added = set()
        for feature in current.keys():
            data_avg[stage][feature] = []

            for i in range(np.max(groups[stage])):
                data_0 = current[feature][(groups[stage] == i) & (labels[stage] == 0)]
                data_1 = current[feature][(groups[stage] == i) & (labels[stage] == 1)]

                if len(data_0) == 0 or len(data_1) == 0:
                    print(f'Dropping subject {names[stage][i]} (empty class)')
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

    age_suffix = ''
    if MIN_AGE >= 0:
        age_suffix += f'_age_f{MIN_AGE}'
    if MAX_AGE >= 0:
        if MIN_AGE < 0:
            age_suffix += f'_age_t{MAX_AGE}'
        else:
            age_suffix += f'-t{MAX_AGE}'

    with open(os.path.join(PATH, f'data{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(data, file)

    with open(os.path.join(PATH, f'labels{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(labels, file)

    with open(os.path.join(PATH, f'groups{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(groups, file)

    with open(os.path.join(PATH, f'data_avg{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(data_avg, file)

    with open(os.path.join(PATH, f'labels_avg{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(labels_avg, file)

    with open(os.path.join(PATH, f'groups_avg{age_suffix}.pickle'), 'wb') as file:
        pickle.dump(groups_avg, file)
