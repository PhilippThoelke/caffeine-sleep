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

if len(sys.argv) > 2:
    CAF_DOSE = sys.argv[2]

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


assert os.path.exists(RESULTS_PATH), "Please make sure the results path exists."


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
    with open(os.path.join(DATA_PATH, "data_avg.pickle"), "rb") as file:
        data = pickle.load(file)
    with open(os.path.join(DATA_PATH, "labels_avg.pickle"), "rb") as file:
        labels = pickle.load(file)
    with open(os.path.join(DATA_PATH, "groups_avg.pickle"), "rb") as file:
        groups = pickle.load(file)

    assert len(sys.argv) > 1, (
        f"please provide the index of the classifier to train as a command line argument: "
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
        RESULTS_PATH, f"singleML{CAF_DOSE}", f"scores_{CLASSIFIER}.pickle"
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


if __name__ == "__main__":
    main()
