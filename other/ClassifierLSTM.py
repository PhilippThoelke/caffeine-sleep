import os
import re
import sys
import glob
import datetime
import numpy as np
from sklearn import model_selection
from keras.wrappers import scikit_learn
from keras import models, layers, callbacks
from Loader import load_data, normalize_data


def get_model(optimizer='adam', loss='binary_crossentropy', input_shape=(5120, 20)):
    model = models.Sequential()
    model.add(layers.LSTM(units=32, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(units=16))
    model.add(layers.Dense(2, activation='softmax'))

    print(model.summary())

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
    
    x = x[:,::10]

    print('Data shape:', x.shape)

    train_scores = []
    test_scores = []

    iteration = 1

    if cv.lower() == 'kfold':
        cross_validation = model_selection.GroupKFold(n_splits=cv_arg)
    elif cv.lower() == 'leavepout':
        cross_validation = model_selection.LeavePGroupsOut(n_groups=cv_arg)
    else:
        print(f'Unknown validation method {cv}')
        return

    if electrode is None:
        timestamp = datetime.datetime.now().strftime(f'lstm_%Y-%m-%d_%H-%M-%S')
    else:
        timestamp = datetime.datetime.now().strftime(f'lstm_elec{electrode}_%Y-%m-%d_%H-%M-%S')

    for train, test in cross_validation.split(x, y, groups):
        print(f'Training model on fold {iteration} of {cv_arg}...', flush=True)

        tensorboard = callbacks.TensorBoard(log_dir=os.path.join(logdir, os.path.join(timestamp, f'{iteration}')))

        clf = scikit_learn.KerasClassifier(get_model,
                                           input_shape=x.shape[1:],
                                           epochs=epochs,
                                           callbacks=[tensorboard],
                                           validation_data=(x[test], y[test]),
                                           batch_size=128,
                                           verbose=2)

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
         epochs=300,
         cv='kfold',
         cv_arg=10,
         electrode=electrode)
