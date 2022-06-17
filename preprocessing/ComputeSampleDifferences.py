import os
import glob
import pickle
import numpy as np

# caffeine dose: 200 or 400
CAF_DOSE = 200
# directory where raw EEG files are stored
RAW_FILES = f"data/raw_eeg{CAF_DOSE}"
# directory where the result should be saved
DIFFERENCE_FILES = "data/"


def get_files(stage):
    files = glob.glob(os.path.join(RAW_FILES, f"*{stage}*"))

    files_subjects = {}
    for file in files:
        subject = file.split(os.sep)[-1].split("_")[0]
        if subject not in files_subjects:
            files_subjects[subject] = []
        files_subjects[subject].append(file)
    return files_subjects


def get_sample_count(files_subjects):
    samples_subjects = {}
    for subject, files in files_subjects.items():
        samples_subjects[subject] = 0
        for file in files:
            samples_subjects[subject] += np.load(file).shape[0]
    return samples_subjects


def get_difference(awa_samples, awsl_samples):
    drop_counts = {}
    for subject in awa_samples.keys():
        if subject in awsl_samples:
            drop_counts[subject] = awa_samples[subject] - awsl_samples[subject]
        else:
            drop_counts[subject] = awa_samples[subject]
    return drop_counts


print("Gathering file names...")
awa_files = get_files("AWA")
awsl_files = get_files("AWSL")

print("Extracting sample counts...")
awa_samples = get_sample_count(awa_files)
awsl_samples = get_sample_count(awsl_files)

print("Evaluating sample differences...")
difference = get_difference(awa_samples, awsl_samples)

path = os.path.join(DIFFERENCE_FILES, f"sample_difference{CAF_DOSE}.pickle")
print(f'Saving difference to "{path}"...')
with open(path, "wb") as file:
    pickle.dump(difference, file)
