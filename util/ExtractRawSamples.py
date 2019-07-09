import sys
sys.path.append('..')

import os
import glob
import numpy as np
import pandas as pd
from caffeine import EEGProcessing

CAF_DOSE = 400
BATCH_SIZE = 64
STAGE = 'AWSL'

DATA_PATH = 'E:\\Cafeine_data\\CAF_{dose}\\EEG_data'.format(dose=CAF_DOSE)
SUBJECTS_PATH = 'E:\\Cafeine_data\\CAF_{dose}_Inventaire.csv'.format(dose=CAF_DOSE)
SAVE_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\raw_eeg{dose}'.format(dose=CAF_DOSE)

subject_ids = pd.read_csv(SUBJECTS_PATH, index_col=0)[['Subject_id', 'CAF']]
for i, (subject_id, label) in subject_ids.iterrows():
    if CAF_DOSE == 200:
        if label == 'Y':
            label = 'CAF'
        else:
            label = 'PLAC'
    elif CAF_DOSE == 400:
        if label == 1:
            label = 'CAF'
        else:
            label = 'PLAC'

    print(f'----------------------------- NEW SUBJECT: {subject_id} -----------------------------')
    done_paths = glob.glob(os.path.join(SAVE_PATH, f'*_{STAGE}_*'))
    done_subjects = [curr.split('\\')[-1].split('_')[0] for curr in done_paths]
    if subject_id in done_subjects:
        print('Done already, moving on...')
        continue

    print('Loading data...')
    eeg_path = os.path.join(DATA_PATH, subject_id, 'EEG_data_clean.npy')
    hyp_path = os.path.join(DATA_PATH, subject_id, 'hyp_clean.npy')
    eeg, hyp = EEGProcessing.load_data(eeg_path, hyp_path, dtype=np.float32)

    print('Extracting sleep stages...')
    eeg_stages = EEGProcessing.extract_sleep_stages(eeg, hyp)
    del eeg, hyp

    print('Saving data batches...')
    for i in range(int(np.ceil(eeg_stages[STAGE].shape[2] / BATCH_SIZE))):
        batch = eeg_stages[STAGE].T[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        data_path = os.path.join(SAVE_PATH, f'{subject_id}_{STAGE}_{BATCH_SIZE}_{i}_{label}.npy')
        np.save(data_path, batch)
    del eeg_stages