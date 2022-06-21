import os
import sys
import pickle
import numpy as np
from joblib import Parallel, delayed
from sklearn import model_selection, ensemble


# caffeine dose: 200 or 400 (can be overwritten using command line arguments)
CAF_DOSE = 200
# number of random forests that should be trained for robustnes
iterations = 1000
# sleep stage index (can be overwritten using command line arguments)
stage_index = 0

# allow to set the caffeine dose with a command line argument
if len(sys.argv) > 1:
    CAF_DOSE = sys.argv[1]

# allow to set the sleep stage with a command line argument
if len(sys.argv) > 2:
    stage_index = int(sys.argv[2])

DATA_PATH = f"data/Features{CAF_DOSE}/Combined"
RESULTS_PATH = f"results/randomForest_avg{CAF_DOSE}"

STAGES = ["AWSL", "NREM", "REM"]
STAGE = STAGES[stage_index]

# load data
with open(os.path.join(DATA_PATH, "data_avg.pickle"), "rb") as file:
    data = pickle.load(file)[STAGE]

# load labels
with open(os.path.join(DATA_PATH, "labels_avg.pickle"), "rb") as file:
    y = pickle.load(file)[STAGE]

# load group vectors
with open(os.path.join(DATA_PATH, "groups_avg.pickle"), "rb") as file:
    groups = pickle.load(file)[STAGE]

# generate a feature name vector WITH SpecPermEn
feature_names = np.concatenate(
    [[feature + "-" + str(i) for i in range(20)] for feature in data.keys()]
)

print(f"Multi feature random forest: CAF{CAF_DOSE}, stage {STAGE}")

x = []
# create a sample matrix from the data dict
for feature in data.keys():
    x.append(data[feature])
x = np.concatenate(x, axis=1)

print(f"Found {len(x)} samples")

# leave P groups out as testing data
cv = model_selection.LeavePGroupsOut(n_groups=5)
# make a list out of the data split generator for random access
cv_split = list(cv.split(x, y, groups))


def train(train, test):
    # initialize a random forest
    clf = ensemble.RandomForestClassifier()

    # generate a parameter dict for the random forest during grid search
    params = {
        "n_estimators": [75, 100, 150, 200],
        "max_depth": [None, 50, 75],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 5, 10],
        "bootstrap": [True, False],
        "n_jobs": [-1],
    }

    # generate a K-fold split of the training subset for cross validation during the grid search
    kfold_inner = model_selection.GroupKFold(n_splits=7)
    inner_cross_validation_split = kfold_inner.split(x[train], y[train], groups[train])

    # run grid search
    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=params,
        cv=inner_cross_validation_split,
        iid=False,
        refit=True,
        n_jobs=-1,
    )
    grid_search.fit(x[train], y[train], groups[train])

    return (
        grid_search.best_estimator_.feature_importances_,
        grid_search.best_estimator_.score(x[test], y[test]),
    )


# perform grid search and training n times
perm = np.random.permutation(len(cv_split))
results = Parallel(n_jobs=-1)(
    delayed(train)(*cv_split[perm[i]]) for i in range(iterations)
)

# extract importances and accuracies for the n random forests
importances = [result[0] for result in results]
scores = [result[1] for result in results]

print("mean score:", np.mean(scores), "\n", flush=True)

# save the trained estimators
with open(os.path.join(RESULTS_PATH, f"importances-{STAGE}.pickle"), "wb") as file:
    pickle.dump(importances, file)

# save the testing data corresponding to each of the estimators
with open(os.path.join(RESULTS_PATH, f"scores-{STAGE}.pickle"), "wb") as file:
    pickle.dump(scores, file)

# save the feature name vector
with open(os.path.join(RESULTS_PATH, f"feature_names-{STAGE}.pickle"), "wb") as file:
    pickle.dump(feature_names, file)
