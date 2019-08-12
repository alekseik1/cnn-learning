import warnings
from itertools import islice, permutations
import os
import keras
import numpy as np
from keras.models import Input
from keras import Model


def test_layer(layer, data):
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data)


def extend_bags_permutations(x_bags, labels, total_num=100):
    """
    Shuffles elements in each bag, and then returns the dataset extended by permutations
    :param x_bags: original bags to permutate
    :param labels: labels corresponding to bags
    :param total_num: total numbers of permutations for each bag
    :return: (x_bags_new, y_labels_new) -- extended array of bags and corresponding labels
    """
    result_x, result_y = [], []
    for i in range(len(x_bags)):
        perms = list(islice(permutations(x_bags[i]), total_num))
        result_x.extend(perms)
        result_y.extend([labels[i] for _ in range(len(perms))])
    return np.array(result_x), np.array(result_y)


class SaveCallback(keras.callbacks.Callback):
    """
        Adapted code from: https://github.com/keras-team/keras/blob/master/keras/callbacks.py
        Class for customized Callbacks to correctly save models trained on multiple GPUs
    """

    def __init__(self, model, monitor_variable, verbose=0,
                 save_best_only=True,
                 save_dir='trained',
                 mode='auto', period=1):
        super(SaveCallback, self).__init__()
        self.model_to_save = model
        self.monitor_variable = monitor_variable
        self.verbose = verbose
        # Check if directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.filepath = os.path.join(os.getcwd(), save_dir,
                                     'model_trained.{epoch:02d}-{%s:.2f}.h5' % monitor_variable)
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor_variable or self.monitor_variable.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # print('logs are: {}'.format(logs))
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor_variable)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor_variable, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor_variable, self.best,
                                     current, filepath))
                        self.best = current

                        self.model_to_save.save(filepath, overwrite=True, include_optimizer=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor_variable))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.model_to_save.save(filepath, overwrite=True, include_optimizer=True)