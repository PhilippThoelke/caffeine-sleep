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
AGE_GROUP = -1  # -1: all, 0: up to age 30, 1: from age 30
USE_AVERAGED_FEATURES = False

DATA_PATH = f"data/Features{CAF_DOSE}/Combined"
RESULTS_PATH = f"results/singleML{CAF_DOSE}"

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

# read command line arguments
# argument 1: classifier id (see above list of classifiers)
if len(sys.argv) > 1:
    CLASSIFIER = CLASSIFIERS[int(sys.argv[1])]

# argument 2: caffeine dose (200 or 400)
if len(sys.argv) > 2:
    CAF_DOSE = sys.argv[2]

# argument 3: age group (-1, 0 or 1)
if len(sys.argv) > 3:
    AGE_GROUP = int(sys.argv[3])

if len(sys.argv) > 4:
    USE_AVERAGED_FEATURES = bool(int(sys.argv[4]))

STAGES = ["NREM", "REM"]
BANDS = ["delta", "theta", "alpha", "sigma", "beta", "low gamma"]


def get_classifier(name, **kwargs):
    if name.lower() == "svm":
        return svm.SVC(gamma="auto", **kwargs)
    elif name.lower() == "lda":
        return discriminant_analysis.LinearDiscriminantAnalysis(**kwargs)
    elif name.lower() == "qda":
        return discriminant_analysis.QuadraticDiscriminantAnalysis(**kwargs)
    elif name.lower() == "gradientboosting":
        return ensemble.GradientBoostingClassifier(**kwargs)
    elif name.lower() == "kneighbors":
        return neighbors.KNeighborsClassifier(n_jobs=-1, **kwargs)
    elif name.lower() == "adaboost":
        return ensemble.AdaBoostClassifier(**kwargs)
    elif name.lower() == "gaussianprocess":
        return gaussian_process.GaussianProcessClassifier(n_jobs=-1, **kwargs)
    elif name.lower() == "perceptron":
        return linear_model.Perceptron(n_jobs=-1, **kwargs)


def main():
    # get age suffix for loading the data depending on age group parameter
    age_suffix = ""
    if AGE_GROUP == 0:
        age_suffix = "_age_t30"
    elif AGE_GROUP == 1:
        age_suffix = "_age_f30"
    elif AGE_GROUP != -1:
        raise Exception(f"Unknown age group {AGE_GROUP}")

    feature_suffix = "_avg" if USE_AVERAGED_FEATURES else ""
    with open(
        os.path.join(DATA_PATH, f"data{feature_suffix}{age_suffix}.pickle"), "rb"
    ) as file:
        data = pickle.load(file)
    with open(
        os.path.join(DATA_PATH, f"labels{feature_suffix}{age_suffix}.pickle"), "rb"
    ) as file:
        labels = pickle.load(file)
    with open(
        os.path.join(DATA_PATH, f"groups{feature_suffix}{age_suffix}.pickle"), "rb"
    ) as file:
        groups = pickle.load(file)

    assert os.path.exists(RESULTS_PATH), "Please make sure the results path exists."

    assert len(sys.argv) > 1, (
        f"please provide the index of the classifier to train as a command line argument: "
        f"{dict(zip(range(len(CLASSIFIERS)),CLASSIFIERS))}"
    )

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
                x[np.isnan(x)] = 0

                y = labels[stage]
                g = groups[stage]
                print(f", permutation test on {len(x)} samples", end="")

                # train classifier
                kfold = model_selection.GroupKFold(n_splits=10)
                try:
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
                except ValueError as e:
                    print(f", ERROR: {e}")
                    score = [float("nan"), None, float("nan")]
                scores[stage][feature].append(score)
            print()

    path = os.path.join(RESULTS_PATH, f"scores_{CLASSIFIER}{feature_suffix}{age_suffix}.pickle")
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


if __name__ == "__main__":
    main()
