import sys
sys.path.append('..')

import os
import numpy as np
import pandas as pd
from sklearn import decomposition
from caffeine import EEGProcessing
from scipy import stats, signal

CAF_DOSE = 400
CLEAN_STAGE = 'REM'

SUBJECTS_PATH = 'E:\\Cafeine_data\\CAF_{dose}_Inventaire.csv'.format(dose=CAF_DOSE)
EEG_PATH = 'E:\\Cafeine_data\\CAF_{dose}\\EEG_data\\'.format(dose=CAF_DOSE)
EOG_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\EOG{dose}'.format(dose=CAF_DOSE)
FEATURES_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\Features{dose}'.format(dose=CAF_DOSE)

def filter_stage(eeg, eog, remove_count=3):
    eog_channels = eog.shape[0]
    epoch_length = eeg.shape[1]
    epoch_count = eeg.shape[2]

    # concatenate the eog columns to the eeg data
    combined = np.concatenate([eeg, eog], axis=0)
    # concatenate all epochs
    data = np.concatenate([combined[:,:,i] for i in range(combined.shape[2])], axis=1).T
    # z-transform the data for PCA
    data = stats.zscore(data, axis=1)

    # apply PCA
    pca = decomposition.PCA()
    projected = pca.fit_transform(data)

    # get correlation between each EOG channel and each PCA component
    correlation = np.empty((projected.shape[1], eog_channels))
    for i in range(projected.shape[1]):
        for channel in range(eog_channels):
            correlation[i,channel] = np.corrcoef(data[:,-(channel+1)], projected[:,i])[0,1]

    # find indices of components with highest correlation in all EOG channels
    indices_sorted = np.argsort(correlation.sum(axis=1).flatten())[::-1]
    # remove PCA channels with highest EOG correlation
    for index in indices_sorted[:remove_count]:
        projected[:,index] = 0

    # project back to original space
    reconstructed = pca.inverse_transform(projected)

    # reshape the data into the original epoch shape
    reconstructed_epochs = np.empty(combined.shape)
    for epoch in range(epoch_count):
        reconstructed_epochs[:,:,epoch] = reconstructed[epoch*epoch_length:(epoch+1)*epoch_length].T

    return reconstructed_epochs[:-eog_channels], reconstructed_epochs[-eog_channels:]

def clean():
    subjects = pd.read_csv(SUBJECTS_PATH, index_col=0)
    for subject_id in subjects['Subject_id']:
        print(f'--------------------------- NEW SUBJECT: {subject_id} ---------------------------')
        path = os.path.join(FEATURES_PATH, subject_id, 'PSD', f'PSD_C{CLEAN_STAGE}.npy')
        if os.path.exists(path):
            print('Done already, moving on...')
            continue

        print('Loading EEG data...')
        eeg_path = os.path.join(EEG_PATH, subject_id, 'EEG_data_clean.npy')
        eeg = np.load(eeg_path)

        print('Loading EOG data...')
        eog_path = os.path.join(EOG_PATH, subject_id + '_EOG.npy')
        eog = np.load(eog_path)

        if eeg.shape[2] != eog.shape[2]:
            print(f'EEG and EOG have different epoch counts ({eeg.shape[2]} and {eog.shape[2]}), moving on...')
            continue

        print('Loading hypnogram...')
        hypnogram_path = os.path.join(EEG_PATH, subject_id, 'hyp_clean.npy')
        hypnogram = np.load(hypnogram_path)

        eeg_stages = EEGProcessing.extract_sleep_stages(eeg, hypnogram)
        eog_stages = EEGProcessing.extract_sleep_stages(eog, hypnogram)
        del eeg, eog

        print(f'Filtering {CLEAN_STAGE} sleep stage...')
        eeg_filtered, _ = filter_stage(eeg_stages[CLEAN_STAGE], eog_stages[CLEAN_STAGE])
        del eeg_stages, eog_stages

        print('Computing PSD...')
        psd = EEGProcessing.power_spectral_density(eeg_filtered)
        np.save(path, psd)

if __name__ == '__main__':
    clean()