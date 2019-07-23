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
        return svm.SVC(**params)
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
                params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3], 'gamma': ['auto', 'scale']}
            elif CLASSIFIER.lower() == 'lda':
                params = {'solver': ['svd', 'lsqr', 'eigen']}
            elif CLASSIFIER.lower() == 'qda':
                params = {'reg_param': [0, 0.25, 0.5]}
            elif CLASSIFIER.lower() == 'gradientboosting':
                params = {'n_estimators': [50, 100]}
            elif CLASSIFIER.lower() == 'adaboost':
                params = {'n_estimators': [50, 100]}
            elif CLASSIFIER.lower() == 'kneighbors':
                params = {'weights': ['uniform', 'distance']}
            elif CLASSIFIER.lower() == 'gaussianprocess':
                params = {'n_restarts_optimizer': [0, 50, 100], 'max_iter_predict': [100, 300]}
            elif CLASSIFIER.lower() == 'perceptron':
                params = {'penalty': [None, 'l2', 'l1', 'elasticnet'], 'fit_intercept': [True, False]}

            x = data[stage][feature][:,electrode].reshape((-1, 1))
            y = labels[stage]
            g = groups[stage]

            # perform grid search
            leave_p_out = model_selection.LeavePGroupsOut(n_groups=len(np.unique(g)) // 4 * 3)
            test_split = leave_p_out.split(X=x, y=y, groups=g)
            train_indices, test_indices = next(test_split)

            train_x = x[train_indices]
            train_y = y[train_indices]
            train_groups = g[train_indices]

            print(f' Train: {len(train_x)}, test: {len(x) - len(train_x)}', end='')

            kfold_grid = model_selection.GroupKFold(n_splits=5)
            grid_search = model_selection.GridSearchCV(estimator=get_classifier(CLASSIFIER),
                                                       param_grid=params,
                                                       cv=kfold_grid.split(X=train_x, y=train_y, groups=train_groups),
                                                       iid=False,
                                                       n_jobs=-1)
            grid_search.fit(X=train_x,
                            y=train_y,
                            groups=train_groups)

            # train classifier
            kfold = model_selection.GroupKFold(n_splits=10)
            score = model_selection.permutation_test_score(estimator=get_classifier(CLASSIFIER, params=grid_search.best_params_),
                                                           n_permutations=1000,
                                                           X=x,
                                                           y=y,
                                                           groups=g,
                                                           cv=kfold.split(X=x, y=y, groups=g),
                                                           n_jobs=-1)
            print(f' score: {score[0]}, pvalue: {score[2]}')
            scores[stage][feature].append(score)
        print()

with open(os.path.join(RESULTS_PATH, f'scores_single{CAF_DOSE}', f'scores_{CLASSIFIER}.pickle'), 'wb') as file:
    pickle.dump(scores, file)

all_scores = [[[elec[0] for elec in ft] for ft in stage.values()] for stage in scores.values()]
vmin = np.min(all_scores)
vmax = np.max(all_scores)

print(f'Min accuracy: {vmin * 100:.2f}%')
print(f'Max accuracy: {vmax * 100:.2f}%')
print(f'Mean accuracy: {np.mean(all_scores) * 100:.2f}%')

plot_rows = 2
plot_cols = 5
colormap = 'coolwarm'

for stage in STAGES:
    plt.figure(figsize=(18, 5))
    plt.suptitle(stage, y=1.05, fontsize=20)

    all_scores = [[elec[0] for elec in ft] for ft in scores[stage].values()]
    vmin = np.min(all_scores)
    vmax = np.max(all_scores)

    subplot_index = 1
    axes = []
    for feature in scores[stage].keys():
        curr_acc = np.array([score[0] for score in scores[stage][feature]])
        curr_sig = np.array([score[2] for score in scores[stage][feature]])

        ax = plt.subplot(plot_rows, plot_cols, subplot_index)
        axes.append(ax)
        plt.title(feature)
        mask = curr_sig < SIGNIFICANT_P
        viz.plot_topomap(curr_acc, sensor_pos, mask=mask, cmap=colormap, vmin=vmin, vmax=vmax, contours=False, show=False)
        subplot_index += 1

    norm = colors.Normalize(vmin=vmin,vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes, shrink=0.95, aspect=15)
    plt.savefig(os.path.join(RESULTS_PATH, f'figures_single{CAF_DOSE}', f'{CLASSIFIER}_DA_{stage}.png'))
