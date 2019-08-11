import os
import sys
import pickle
import numpy as np
from mne import viz
from scipy import io
from matplotlib import colors, pyplot as plt
from sklearn import model_selection, ensemble, svm, discriminant_analysis, neighbors, linear_model, gaussian_process

CAF_DOSE = 200
SIGNIFICANT_P = 0.05

if len(sys.argv) > 2:
    CAF_DOSE = sys.argv[2]

CLASSIFIERS = ['SVM', 'LDA', 'Perceptron', 'GaussianProcess']

DATA_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/Features{dose}/Combined'.format(dose=CAF_DOSE)
PROJECT_PATH = '/home/pthoelke/caffeine'
RESULTS_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/results'

#DATA_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\Features{dose}\\Combined'.format(dose=CAF_DOSE)
#PROJECT_PATH = '..\\data' # path to where the EEG sensor position file is stored
#RESULTS_PATH = '..\\results'

STAGES = ['AWSL', 'NREM', 'REM']
BANDS = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'low gamma']

sensor_pos = io.loadmat(os.path.join(PROJECT_PATH, 'Coo_caf'))['Cor'].T
sensor_pos = np.array([sensor_pos[1], sensor_pos[0]]).T

with open(os.path.join(DATA_PATH, 'data_avg.pickle'), 'rb') as file:
    data = pickle.load(file)
with open(os.path.join(DATA_PATH, 'labels_avg.pickle'), 'rb') as file:
    labels = pickle.load(file)
with open(os.path.join(DATA_PATH, 'groups_avg.pickle'), 'rb') as file:
    groups = pickle.load(file)

def get_classifier(name, params={}):
    if CLASSIFIER.lower() == 'svm':
        return svm.SVC(gamma='auto', **params)
    elif CLASSIFIER.lower() == 'lda':
        return discriminant_analysis.LinearDiscriminantAnalysis(**params)
    elif CLASSIFIER.lower() == 'qda':
        return discriminant_analysis.QuadraticDiscriminantAnalysis(**params)
    elif CLASSIFIER.lower() == 'gradientboosting':
        return ensemble.GradientBoostingClassifier(**params)
    elif CLASSIFIER.lower() == 'kneighbors':
        return neighbors.KNeighborsClassifier(n_jobs=-1, **params)
    elif CLASSIFIER.lower() == 'adaboost':
        return ensemble.AdaBoostClassifier(**params)
    elif CLASSIFIER.lower() == 'gaussianprocess':
        return gaussian_process.GaussianProcessClassifier(n_jobs=-1, **params)
    elif CLASSIFIER.lower() == 'perceptron':
        return linear_model.Perceptron(max_iter=1000, tol=1e-3, n_jobs=-1, **params)

clf_id = int(sys.argv[1])
CLASSIFIER = CLASSIFIERS[clf_id]

print(f'Decoding accuracy average, {CLASSIFIER} on CAF {CAF_DOSE}')

scores = {}
for stage in STAGES:
    scores[stage] = {}
    print(f'Sleep stage {stage}')
    for feature in data[stage].keys():
        scores[stage][feature] = []
        for electrode in range(20):
            print(f'   {CLASSIFIER} on {feature}, elec {electrode + 1}', end='', flush=True)

            if CLASSIFIER.lower() == 'svm':
                params = {}
            elif CLASSIFIER.lower() == 'lda':
                params = {}
            elif CLASSIFIER.lower() == 'qda':
                params = {}
            elif CLASSIFIER.lower() == 'gradientboosting':
                params = {}
            elif CLASSIFIER.lower() == 'adaboost':
                params = {}
            elif CLASSIFIER.lower() == 'kneighbors':
                params = {}
            elif CLASSIFIER.lower() == 'gaussianprocess':
                params = {}
            elif CLASSIFIER.lower() == 'perceptron':
                params = {}

            x = data[stage][feature][:,electrode].reshape((-1, 1))
            y = labels[stage]
            g = groups[stage]
            print(f', permutation test on {len(x)} samples', end='')

            # train classifier
            kfold = model_selection.GroupKFold(n_splits=10)
            score = model_selection.permutation_test_score(estimator=get_classifier(CLASSIFIER, params=params),
                                                           n_permutations=1000,
                                                           X=x,
                                                           y=y,
                                                           groups=g,
                                                           cv=kfold.split(X=x, y=y, groups=g),
                                                           n_jobs=-1)
            print(f', score: {score[0]}, pvalue: {score[2]}')
            scores[stage][feature].append(score)
        print()

with open(os.path.join(RESULTS_PATH, f'scores_single_noGrid{CAF_DOSE}', f'scores_{CLASSIFIER}.pickle'), 'wb') as file:
    pickle.dump(scores, file)

all_scores = [[[elec[0] for elec in ft] for ft in stage.values()] for stage in scores.values()]
vmin = np.min(all_scores)
vmax = np.max(all_scores)

print(f'Min accuracy: {vmin * 100:.2f}%')
print(f'Max accuracy: {vmax * 100:.2f}%')
print(f'Mean accuracy: {np.mean(all_scores) * 100:.2f}%')
