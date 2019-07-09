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

#DATA_PATH = 'C:\\Users\\Philipp\\Documents\\Caffeine\\Features{dose}\\Combined'.format(dose=CAF_DOSE)
#RESULTS_PATH = 'C:\\Users\\Philipp\\GoogleDrive\\Caffeine\\results\\randomForestAll{dose}'.format(dose=CAF_DOSE)

DATA_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/Features{dose}/Combined'.format(dose=CAF_DOSE)
RESULTS_PATH = '/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/results/randomForestAll{dose}'.format(dose=CAF_DOSE)

STAGES = ['AWA', 'AWSL', 'NREM', 'REM']
STAGE = STAGES[stage_index]

# boolean indicating whether SpecPermEn and SpecSampEn should be included
INCLUDE_SPEC = False

# load data
with open(os.path.join(DATA_PATH, 'data.pickle'), 'rb') as file:
    data = pickle.load(file)[STAGE]

# load labels
with open(os.path.join(DATA_PATH, 'labels.pickle'), 'rb') as file:
    y = pickle.load(file)[STAGE]

# load group vectors
with open(os.path.join(DATA_PATH, 'groups.pickle'), 'rb') as file:
    groups = pickle.load(file)[STAGE]

if INCLUDE_SPEC:
    # generate a feature name vector WITH SpecPermEn and SpecSampEn
    feature_names = np.concatenate([[feature + '-' + str(i) for i in range(20)] for feature in data.keys()])
else:
    # generate a feature name vector WITHOUT SpecPermEn and SpecSampEn
    feature_names = np.concatenate([[feature + '-' + str(i) for i in range(20)] for feature in data.keys() if 'Perm' not in feature and 'SpecSamp' not in feature])

print(f'Multi feature random forest, include spec perm and spec samp: {INCLUDE_SPEC}, CAF{CAF_DOSE}, stage {STAGE}')

x = []
# create a sample matrix from the data dict
for feature in data.keys():
    if not INCLUDE_SPEC:
        if 'Perm' in feature or 'SpecSamp' in feature:
            # skip SpecPermEn and SpecSampEn if INCLUDE_SPEC is false
            continue
    x.append(data[feature])
x = np.concatenate(x, axis=1)

# initialize data structures for the results
estimator_dict = {}
testing_data_dict = {}

testing_data = []
estimators = []
avg_score = []

print(f'Found {len(x)} samples')

# leave P groups out as testing data
cv = model_selection.LeavePGroupsOut(n_groups=4)
# make a list out of the data split generator for random access
cv_split = list(cv.split(x, y, groups))

counter = 0
# iterate through a random permutation of the cross validation split indices
for i in np.random.permutation(len(cv_split)):
    # get current train-test split
    train, test = cv_split[i]

    if counter % 25 == 0:
        print(f'{STAGE} iteration {counter}/{iterations}', flush=True)
    if counter >= iterations:
        # iteration count has been reached, break out of the loop
        break

    # initialize a random forest
    clf = ensemble.RandomForestClassifier(n_jobs=-1)

    # generate a parameter dict for the random forest during grid search
    params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'criterion': ['gini'],
        'max_features': ['auto'],
        'min_samples_leaf': [1],
        'class_weight': ['balanced_subsample'],
        'bootstrap': [True]
    }

    # generate a K-fold split of the training subset for cross validation during the grid search
    kfold_inner = model_selection.GroupKFold(n_splits=10)
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

    # save the current testing data subset, the current trained estimator and its score on the test set
    testing_data.append((x[test], y[test]))
    estimators.append(grid_search.best_estimator_)
    avg_score.append(grid_search.best_estimator_.score(x[test], y[test]))
    counter += 1

testing_data_dict = testing_data
estimator_dict = estimators

print('mean score:', np.mean(avg_score), '\n', flush=True)

# save the trained estimators
with open(os.path.join(RESULTS_PATH, f'estimators-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(estimator_dict, file)

# save the testing data corresponding to each of the estimators
with open(os.path.join(RESULTS_PATH, f'testing_data-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(testing_data_dict, file)

# save the feature name vector
with open(os.path.join(RESULTS_PATH, f'feature_names-{STAGE}.pickle'), 'wb') as file:
    pickle.dump(feature_names, file)
