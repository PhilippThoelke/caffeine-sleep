import os
import sys
import pickle
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed
from sklearn import model_selection, ensemble


# caffeine dose: 200 or 400 (can be overwritten using command line arguments)
CAF_DOSE = 200
# number of random forests that should be trained for robustnes
ITERATIONS = 1000
# sleep stage index (can be overwritten using command line arguments)
STAGE_INDEX = 0
# -1: all, 0: up to age 30, 1: from age 30 (can be overwritten using command line arguments)
AGE_GROUP = -1

# allow to set the caffeine dose with a command line argument
if len(sys.argv) > 1:
    CAF_DOSE = sys.argv[1]

# allow to set the sleep stage with a command line argument
if len(sys.argv) > 2:
    STAGE_INDEX = int(sys.argv[2])

# allow to set the age group with a command line argument
if len(sys.argv) > 3:
    AGE_GROUP = int(sys.argv[3])

DATA_PATH = f"data/Features{CAF_DOSE}/Combined"
RESULTS_PATH = f"results/multiML{CAF_DOSE}"

STAGES = ["AWSL", "NREM", "REM"]
STAGE = STAGES[STAGE_INDEX]

assert os.path.exists(RESULTS_PATH), "Please make sure the results path exists."

# get age suffix for loading the data depending on age group parameter
age_suffix = ""
if AGE_GROUP == 0:
    age_suffix = "_age_t30"
elif AGE_GROUP == 1:
    age_suffix = "_age_f30"
elif AGE_GROUP != -1:
    raise Exception(f"Unknown age group {AGE_GROUP}")

# load data
with open(os.path.join(DATA_PATH, f"data_avg{age_suffix}.pickle"), "rb") as file:
    data = pickle.load(file)[STAGE]

# load labels
with open(os.path.join(DATA_PATH, f"labels_avg{age_suffix}.pickle"), "rb") as file:
    y = pickle.load(file)[STAGE]

# load group vectors
with open(os.path.join(DATA_PATH, f"groups_avg{age_suffix}.pickle"), "rb") as file:
    groups = pickle.load(file)[STAGE]

# generate a feature name vector WITH SpecPermEn
feature_names = np.concatenate(
    [[feature + "-" + str(i) for i in range(20)] for feature in data.keys()]
)

print(
    f"Multi feature random forest: CAF{CAF_DOSE}, stage {STAGE}, age group {AGE_GROUP}"
)

x = []
# create a sample matrix from the data dict
for feature in data.keys():
    x.append(data[feature])
x = np.concatenate(x, axis=1)
# replace NaNs by 0
x[np.isnan(x)] = 0

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
    delayed(train)(*cv_split[perm[i]]) for i in trange(ITERATIONS)
)

# extract importances and accuracies for the n random forests
importances = [result[0] for result in results]
scores = [result[1] for result in results]

print("mean score:", np.mean(scores), "\n", flush=True)

# save the trained estimators
with open(
    os.path.join(RESULTS_PATH, f"importances-{STAGE}{age_suffix}.pickle"), "wb"
) as file:
    pickle.dump(importances, file)

# save the testing data corresponding to each of the estimators
with open(
    os.path.join(RESULTS_PATH, f"scores-{STAGE}{age_suffix}.pickle"), "wb"
) as file:
    pickle.dump(scores, file)

# save the feature name vector
with open(
    os.path.join(RESULTS_PATH, f"feature_names-{STAGE}{age_suffix}.pickle"), "wb"
) as file:
    pickle.dump(feature_names, file)
