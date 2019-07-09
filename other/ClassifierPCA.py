import os
import re
import sys
import glob
import datetime
import numpy as np
from keras.wrappers import scikit_learn
from tensorflow.python.client import device_lib
from sklearn import model_selection, decomposition
from keras import models, layers, callbacks, backend
from Loader import load_data, normalize_data, apply_pca


def get_model(optimizer='adam', loss='binary_crossentropy', input_shape=(5120, 20)):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(rate=0.8))
    model.add(layers.Dense(16, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(rate=0.75))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def main(data_path, model_folder=None, epochs=100, electrode=None, cv='kfold', cv_arg=10, logdir='logs'):
    x_total, y, groups = load_data(data_path)

    if electrode is None:
        x = x_total
    else:
        x = x_total[:,:,electrode][:,:,None]

    del x_total

    normalize_data(x)
    x = apply_pca(x, n_components=32)
    x = np.reshape(x, (x.shape[0], -1))

    print('Data shape:', x.shape)

    gpus = backend.tensorflow_backend._get_available_gpus()
    print(f'Available GPUs ({len(gpus)}):')
    [print(gpu) for gpu in gpus]

    print('\nTensorflow has found these devices:')
    [print(device) for device in device_lib.list_local_devices()]

    train_scores = []
    test_scores = []

    iteration = 1

    if cv.lower() == 'kfold':
        cross_validation = model_selection.GroupKFold(n_splits=cv_arg, shuffle=True)
    elif cv.lower() == 'leavepout':
        cross_validation = model_selection.LeavePGroupsOut(n_groups=cv_arg)
    else:
        print(f'Unknown validation method {cv}')
        return

    timestamp = datetime.datetime.now().strftime(f'pca_elec{electrode}_%Y-%m-%d_%H-%M-%S')

    for train, test in cross_validation.split(x, y, groups):
        print(f'======================== Training model number {iteration}, (cross validation argument: {cv_arg}) ========================', flush=True)

        tensorboard = callbacks.TensorBoard(log_dir=os.path.join(logdir, os.path.join(timestamp, f'{iteration}')))

        clf = scikit_learn.KerasClassifier(get_model,
                                           input_shape=x.shape[1:],
                                           epochs=epochs,
                                           callbacks=[tensorboard],
                                           validation_data=(x[test], y[test]),
                                           batch_size=128,
                                           verbose=2)

        print(clf.model.summary())

        clf.fit(x[train], y[train])

        train_scores.append(clf.score(x[train], y[train]))
        test_scores.append(clf.score(x[test], y[test]))

        if model_folder is not None:
            folder = os.path.join(model_folder, timestamp)
            if not os.path.exists(folder):
                os.mkdir(folder)
            clf.model.save(os.path.join(folder, f'model_it{iteration}.ht'))

        iteration += 1

    if electrode is None:
        print(f'Mean training score:', np.mean(train_scores))
        print(f'Mean testing score:', np.mean(test_scores))
    else:
        print(f'Electrode {electrode} mean training score:', np.mean(train_scores))
        print(f'Electrode {electrode} mean testing score:', np.mean(test_scores))


if __name__ == '__main__':
    electrode = None
    if len(sys.argv) > 1:
        electrode = int(sys.argv[1])

    main(data_path='/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/raw_200_test',
         model_folder='/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/models',
         logdir='/home/pthoelke/projects/def-kjerbi/pthoelke/caffeine/logs',
         epochs=200,
         cv='kfold',
         cv_arg=10,
         electrode=electrode)
