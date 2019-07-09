import sys
sys.path.append('..')

import os
import re
import glob
import numpy as np
from caffeine.EEGProcessing import sample_entropy, power_spectral_density
from joblib import Parallel, delayed

CAF_DOSE = 200
DATA_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\raw_eeg{dose}'.format(dose=CAF_DOSE)
ENTROPY_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\sample_entropy{dose}'.format(dose=CAF_DOSE)

STAGE = 'NREM'
DIMENSION = 10
TOLERANCE = 0.2

def execute(index, file):
    info = file.split(os.sep)[-1].split('.')[0].split('_')

    current = np.load(file)
    psd = power_spectral_density(current.T, bands=False)

    samp_en = np.empty((current.shape[0], DIMENSION, current.shape[2]))
    spec_samp_en = np.empty((psd.shape[1], DIMENSION, psd.shape[0]))

    for epoch in range(current.shape[0]):
        for electrode in range(current.shape[2]):
            samp_en[epoch,:,electrode] = sample_entropy(current[epoch,:,electrode],
                                                        dimension=DIMENSION,
                                                        tolerance=TOLERANCE,
                                                        only_last=False)
            spec_samp_en[epoch,:,electrode] = sample_entropy(psd[electrode,epoch],
                                                             dimension=DIMENSION,
                                                             tolerance=TOLERANCE,
                                                             only_last=False)

    entropy = np.empty((samp_en.shape[0], 2, DIMENSION, samp_en.shape[2]))
    entropy[:,0] = samp_en
    entropy[:,1] = spec_samp_en

    np.save(os.path.join(ENTROPY_PATH, f'{info[0]}_{info[1]}_{info[2]}_{info[3]}_d{DIMENSION}_t{TOLERANCE}_{info[4]}.npy'), entropy)
    print(f'Finished file {index + 1}')


files = glob.glob(os.path.join(DATA_PATH, f'*{STAGE}*'))
print(f'Found {len(files)} files to process')
Parallel(n_jobs=-1)(delayed(execute)(i, file) for i, file in enumerate(files))
