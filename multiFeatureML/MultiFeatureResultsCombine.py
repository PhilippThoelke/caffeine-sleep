import os
import sys
import glob
import pickle
import numpy as np

CAF_DOSE = 200

if len(sys.argv) > 1:
    CAF_DOSE = sys.argv[1]

#RESULTS_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\results\\randomForestAll{dose}'.format(dose=CAF_DOSE)
RESULTS_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/results/randomForestAll{dose}'.format(dose=CAF_DOSE)

def load(paths, data_dict):
    for path in paths:
        stage = path.split('.')[-2].split('-')[-1]
        if stage not in data:
            data_dict[stage] = []
        with open(path, 'rb') as file:
            data_dict[stage].append(pickle.load(file))

data = {}
load(glob.glob(os.path.join(RESULTS_PATH, 'estimators*')), data)
print('Finished loading estimators')
load(glob.glob(os.path.join(RESULTS_PATH, 'testing_data*')), data)
print('Finished loading testing data')
load(glob.glob(os.path.join(RESULTS_PATH, 'feature_names*')), data)
print('Finished loading feature names')

for key in list(data.keys()):
    if len(data[key]) != 3:
        print(f'Dropping stage {key}, not all necessary files were found')
        del data[key]

accuracies = {}
for stage in data.keys():
    accuracies[stage] = np.array([rf.score(*test) for rf, test in zip(*data[stage][:2])])
print('Finished getting accuracies')

importances = {}
for stage in data.keys():
    importances[stage] = np.array([rf.feature_importances_ for rf in data[stage][0]])
print('Finished getting feature importances')

feature_names = {}
for stage in data.keys():
    feature_names[stage] = np.array([names for names in data[stage][2]])
print('Finished getting feature names')

with open(os.path.join(RESULTS_PATH, 'combined_results.pickle'), 'wb') as file:
    pickle.dump((accuracies, importances, feature_names), file)
    print(f'Results saved in "{file.name}"')