import os
import re
import sys
import pickle
import numpy as np
from mne import viz
from scipy import io
from matplotlib import colors, pyplot as plt
from sklearn import model_selection, ensemble, svm, discriminant_analysis, neural_network, linear_model, metrics


CAF_DOSE = 200
electrode = 0
SIGNIFICANT_P = 0.05

electrode = int(sys.argv[1])
CAF_DOSE = sys.argv[2]

CLASSIFIERS = ['SVM', 'LDA', 'RandomForest', 'MultilayerPerceptron']

FEATURE_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/Features{dose}/Combined'.format(dose=CAF_DOSE)
PROJECT_PATH = '/home/pthoelke/caffeine'
RESULTS_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/results'

STAGES = ['AWSL', 'NREM', 'REM']
BANDS = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'low gamma']


sensor_pos = io.loadmat(os.path.join(PROJECT_PATH, 'Coo_caf'))['Cor'].T
sensor_pos = np.array([sensor_pos[1], sensor_pos[0]]).T


with open(os.path.join(FEATURE_PATH, 'data_avg.pickle'), 'rb') as file:
    data = pickle.load(file)
with open(os.path.join(FEATURE_PATH, 'labels_avg.pickle'), 'rb') as file:
    labels = pickle.load(file)
with open(os.path.join(FEATURE_PATH, 'groups_avg.pickle'), 'rb') as file:
    groups = pickle.load(file)


def train_electrode(x, y, g, clf, stage, electrode):
    if clf_name.lower() == 'svm':
        clf = svm.SVC
        params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3], 'gamma': ['auto', 'scale']}
    elif clf_name.lower() == 'lda':
        clf = discriminant_analysis.LinearDiscriminantAnalysis
        params = {'solver': ['svd', 'lsqr', 'eigen']}
    elif clf_name.lower() == 'multilayerperceptron':
        clf = neural_network.MLPClassifier
        params = {'max_iter': [3000], 'hidden_layer_sizes': [(8,), (4,)], 'activation': ['relu', 'tanh', 'logistic'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
    elif clf_name.lower() == 'randomforest':
        clf = ensemble.RandomForestClassifier
        params = {'n_jobs': [-1], 'n_estimators': [10, 20, 40, 60, 100], 'max_depth': [5, 10, 50, None], 'max_features': ['sqrt', 'log2'], 'min_samples_leaf': [1, 2, 5]}

    # randomly take half of the subjects away for the grid search
    half_out = model_selection.LeavePGroupsOut(n_groups=len(np.unique(g)) // 2)
    train_indices, _ = next(half_out.split(x, y, g))
    x_train, y_train, g_train = x[train_indices], y[train_indices], g[train_indices]

    # use KFold cross validation in the grid search
    grid_kfold = model_selection.GroupKFold(n_splits=5)
    grid_search = model_selection.GridSearchCV(estimator=clf(),
                                               param_grid=params,
                                               iid=True,
                                               cv=grid_kfold.split(x_train, y_train, g_train),
                                               n_jobs=-1)
    grid_search.fit(x_train, y_train, g_train)

    kfold = model_selection.GroupKFold(n_splits=10)
    split = list(kfold.split(x, y, g))
    test_indices = [indices[1] for indices in split]

    current = model_selection.permutation_test_score(estimator=clf(**grid_search.best_params_),
                                                     X=x,
                                                     y=y,
                                                     groups=g,
                                                     cv=split,
                                                     n_permutations=1000,
                                                     n_jobs=-1)

    return current[::2], [grid_search.best_estimator_.predict(x[fold]) for fold in test_indices], test_indices


scores = {}
for stage in STAGES:
    scores[stage] = {}

    features = [feature for name, feature in data[stage].items() if not 'SpecPermEn' in name]
    print(f'{stage}: {len(features)} features, {len(features[0])} samples')

    x_all = np.concatenate(features, axis=1)
    y = labels[stage]
    g = groups[stage]

    curr_pred = []

    for clf_name in CLASSIFIERS:
        print(f'    Training {clf_name}...', end='', flush=True)
        results = train_electrode(x_all[:,electrode::20], y, g, clf_name, stage, electrode)

        scores[stage][clf_name] = results[0]
        curr_pred.append(results[1])
        curr_test_indices = results[2]

        print(f'done, score: {scores[stage][clf_name][0]:.3f}', flush=True)

    # get the ensemble prediction forr all folds by averaging class predictions and the rounding
    curr_pred = np.array(curr_pred)
    ensemble_pred = []
    for i in range(curr_pred.shape[1]):
        mean = np.zeros(len(curr_pred[:,i][0]))
        for j in range(len(curr_pred[:,i])):
            mean += curr_pred[:,i][j]
        ensemble_pred.append(np.rint(mean / len(curr_pred[:,i])).astype(int))
    scores[stage]['ensemble'] = np.mean([metrics.accuracy_score(ensemble_pred[i], y[fold]) for i, fold in enumerate(curr_test_indices)])
    print(f'    Ensemble score: {scores[stage]["ensemble"]}')

with open(os.path.join(RESULTS_PATH, f'scores_multi{CAF_DOSE}', f'scores_multi_{electrode}.pickle'), 'wb') as file:
    pickle.dump(scores, file)