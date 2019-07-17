import os
import sys
import pickle
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn import model_selection, ensemble, metrics

CAF_DOSE = 200
iterations = 1000
stage_index = 0

# allow to set the caffeine dose with a command line argument
if len(sys.argv) > 1:
    CAF_DOSE = sys.argv[1]

# allow to set the sleep stage with a command line argument
if len(sys.argv) > 2:
    stage_index = int(sys.argv[2])

DATA_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/Features{dose}/Combined'.format(dose=CAF_DOSE)
RESULTS_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/results/randomForestAll{dose}'.format(dose=CAF_DOSE)

STAGES = ['AWA', 'AWSL', 'NREM', 'REM']
STAGE = STAGES[stage_index]

# boolean indicating whether SpecPermEn should be included
INCLUDE_SPEC_PERM = False

# load data
with open(os.path.join(DATA_PATH, 'data.pickle'), 'rb') as file:
    data = pickle.load(file)[STAGE]

# load labels
with open(os.path.join(DATA_PATH, 'labels.pickle'), 'rb') as file:
    y = pickle.load(file)[STAGE]

# load group vectors
with open(os.path.join(DATA_PATH, 'groups.pickle'), 'rb') as file:
    groups = pickle.load(file)[STAGE]

if INCLUDE_SPEC_PERM:
    # generate a feature name vector WITH SpecPermEn
    feature_names = np.concatenate([[feature + '-' + str(i) for i in range(20)] for feature in data.keys()])
else:
    # generate a feature name vector WITHOUT SpecPermEn
    feature_names = np.concatenate([[feature + '-' + str(i) for i in range(20)] for feature in data.keys() if 'SpecPermEn' not in feature])

print(f'Multi feature random forest, include SpecPermEn: {INCLUDE_SPEC_PERM}, CAF{CAF_DOSE}, stage {STAGE}')

x = []
# create a sample matrix from the data dict
for feature in data.keys():
    if not INCLUDE_SPEC_PERM:
        if 'SpecPermEn' in feature:
            # skip SpecPermEn if INCLUDE_SPEC_PERM is false
            continue
    x.append(data[feature])
x = np.concatenate(x, axis=1)

print(f'Found {len(x)} samples')

# leave P groups out as testing data
cv = model_selection.LeavePGroupsOut(n_groups=5)
# make a list out of the data split generator for random access
cv_split = list(cv.split(x, y, groups))

def train(train, test):
    # initialize a random forest
    clf = ensemble.RandomForestClassifier()

    # generate a parameter dict for the random forest during grid search
    params = {
        'n_estimators': [10, 25, 50, 75],
        'max_depth': [None, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 5, 10],
        'bootstrap': [True, False],
        'n_jobs': [-1]
    }

    # generate a K-fold split of the training subset for cross validation during the grid search
    kfold_inner = model_selection.GroupKFold(n_splits=7)
    inner_cross_validation_split = kfold_inner.split(x[train],
                                                     y[train],
                                                     groups[train])

    # run grid search
    grid_search = model_selection.GridSearchCV(estimator=clf,
                                               param_grid=params,
                                               cv=inner_cross_validation_split,
                                               iid=False,
                                               refit=True,
                                               n_jobs=-1)
    grid_search.fit(x[train], y[train], groups[train])

    return grid_search.best_estimator_.feature_importances_, grid_search.best_estimator_.score(x[test], y[test])

perm = np.random.permutation(len(cv_split))
results = Parallel(n_jobs=-1)(delayed(train)(*cv_split[perm[i]]) for i in range(iterations))

importances = [result[0] for result in results]
scores = [result[1] for result in results]

print('mean score:', np.mean(scores), '\n', flush=True)

# save the trained estimators
with open(os.path.join(RESULTS_PATH, f'importances-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(importances, file)

# save the testing data corresponding to each of the estimators
with open(os.path.join(RESULTS_PATH, f'scores-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(scores, file)

# save the feature name vector
with open(os.path.join(RESULTS_PATH, f'feature_names-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(feature_names, file)
