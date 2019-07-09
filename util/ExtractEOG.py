import sys
sys.path.append('..')

import os
import glob
import numpy as np
from projet_cafeine_Maxine import EdfProcessing

CAF_DOSE = 400

EDF_PATH = 'E:\\Cafeine_data\\CAF_{dose}\\EDF'.format(dose=CAF_DOSE)
EOG_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\EOG{dose}'.format(dose=CAF_DOSE)

EPOCH_LENGTH = 20

edfs = glob.glob(os.path.join(EDF_PATH, '*.edf'))
for path in edfs:
    subject_id = path.split('\\')[-1].split('_')[0]
    print(f'----------------------------- NEW SUBJECT: {subject_id} -----------------------------')

    done = [f.split('\\')[-1].split('_')[0] for f in glob.glob(os.path.join(EOG_PATH, '*.npy'))]
    if subject_id in done:
        print(f'Subject {subject_id} already done')
        continue

    eog_path = os.path.join(EOG_PATH, subject_id + '_EOG.npy')
    np.save(eog_path, np.empty(0))

    xml = path + '.xml'
    hyp, _, art_start, _ = EdfProcessing.extract_markers(xml, False, [], path.split('\\')[-1])

    data, ch, fs = EdfProcessing.extract_data_fromEDF(path)
    eog, channel_names = EdfProcessing.select_data(data, channels_names=ch, data_to_select='EOG')
    segments = EdfProcessing.data_epoching(data=eog,epoch_length=EPOCH_LENGTH,Fs=fs)

    clean_eog, _ = EdfProcessing.remove_artefacts(data=segments, hyp=hyp,
                                                  Art_start=art_start,
                                                  epoch_length=EPOCH_LENGTH)

    np.save(eog_path, clean_eog)
