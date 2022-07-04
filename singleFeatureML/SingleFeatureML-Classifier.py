import os
import sys
import pickle
import numpy as np
from sklearn import (
    model_selection,
    ensemble,
    svm,
    discriminant_analysis,
    neighbors,
    linear_model,
    gaussian_process,
)

CAF_DOSE = 200
SIGNIFICANT_P = 0.05
AGE_GROUP = 1  # -1: all, 0: up to age 30, 1: from age 30

if len(sys.argv) > 2:
    CAF_DOSE = sys.argv[2]

if len(sys.argv) > 3:
    AGE_GROUP = int(sys.argv[3])

CLASSIFIERS = [
    "SVM",
    "LDA",
    "Perceptron",
    "QDA",
    "GradientBoosting",
    "kNeighbors",
    "adaboost",
    "GaussianProcess",
]

DATA_PATH = f"data/Features{CAF_DOSE}/Combined"
RESULTS_PATH = "results"

STAGES = ["AWSL", "NREM", "REM"]
BANDS = ["delta", "theta", "alpha", "sigma", "beta", "low gamma"]

# get age suffix for loading the data depending on age group parameter
age_suffix = ""
if AGE_GROUP == 0:
    age_suffix = "_age_t30"
elif AGE_GROUP == 1:
    age_suffix = "_age_f30"
elif AGE_GROUP != -1:
    raise Exception(f"Unknown age group {AGE_GROUP}")

with open(os.path.join(DATA_PATH, f"data_avg{age_suffix}.pickle"), "rb") as file:
    data = pickle.load(file)
with open(os.path.join(DATA_PATH, f"labels_avg{age_suffix}.pickle"), "rb") as file:
    labels = pickle.load(file)
with open(os.path.join(DATA_PATH, f"groups_avg{age_suffix}.pickle"), "rb") as file:
    groups = pickle.load(file)


def get_classifier(name, **kwargs):
    if CLASSIFIER.lower() == "svm":
        return svm.SVC(gamma="auto", **kwargs)
    elif CLASSIFIER.lower() == "lda":
        return discriminant_analysis.LinearDiscriminantAnalysis(**kwargs)
    elif CLASSIFIER.lower() == "qda":
        return discriminant_analysis.QuadraticDiscriminantAnalysis(**kwargs)
    elif CLASSIFIER.lower() == "gradientboosting":
        return ensemble.GradientBoostingClassifier(**kwargs)
    elif CLASSIFIER.lower() == "kneighbors":
        return neighbors.KNeighborsClassifier(n_jobs=-1, **kwargs)
    elif CLASSIFIER.lower() == "adaboost":
        return ensemble.AdaBoostClassifier(**kwargs)
    elif CLASSIFIER.lower() == "gaussianprocess":
        return gaussian_process.GaussianProcessClassifier(n_jobs=-1, **kwargs)
    elif CLASSIFIER.lower() == "perceptron":
        return linear_model.Perceptron(n_jobs=-1, **kwargs)


assert len(sys.argv) > 1, (
    f"please provide the index of the classifier to train: "
    f"{dict(zip(range(len(CLASSIFIERS)),CLASSIFIERS))}"
)
clf_id = int(sys.argv[1])
CLASSIFIER = CLASSIFIERS[clf_id]

print(f"Decoding accuracy average, {CLASSIFIER} on CAF {CAF_DOSE}")

scores = {}
for stage in STAGES:
    scores[stage] = {}
    print(f"Sleep stage {stage}")
    for feature in data[stage].keys():
        scores[stage][feature] = []
        for electrode in range(20):
            print(
                f"   {CLASSIFIER} on {feature}, elec {electrode + 1}",
                end="",
                flush=True,
            )

            x = data[stage][feature][:, electrode].reshape((-1, 1))
            y = labels[stage]
            g = groups[stage]
            print(f", permutation test on {len(x)} samples", end="")

            # train classifier
            kfold = model_selection.GroupKFold(n_splits=10)
            score = model_selection.permutation_test_score(
                estimator=get_classifier(CLASSIFIER),
                n_permutations=1000,
                X=x,
                y=y,
                groups=g,
                cv=kfold.split(X=x, y=y, groups=g),
                n_jobs=-1,
            )
            print(f", score: {score[0]}, pvalue: {score[2]}")
            scores[stage][feature].append(score)
        print()

path = os.path.join(
    RESULTS_PATH, f"singleML{CAF_DOSE}", f"scores_{CLASSIFIER}{age_suffix}.pickle"
)
with open(path, "wb",) as file:
    pickle.dump(scores, file)

all_scores = [
    [[elec[0] for elec in ft] for ft in stage.values()] for stage in scores.values()
]
vmin = np.min(all_scores)
vmax = np.max(all_scores)

print(f"Min accuracy: {vmin * 100:.2f}%")
print(f"Max accuracy: {vmax * 100:.2f}%")
print(f"Mean accuracy: {np.mean(all_scores) * 100:.2f}%")
