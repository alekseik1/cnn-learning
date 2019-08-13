import os
import warnings
from datetime import datetime
import keras
import numpy as np
from utils import ensure_folder


class SaveCallback(keras.callbacks.Callback):
    """
        Adapted code from: https://github.com/keras-team/keras/blob/master/keras/callbacks.py
        Class for customized Callbacks to correctly save models trained on multiple GPUs
    """

    def __init__(self, model, monitor_variable, verbose=0,
                 save_best_only=True,
                 save_dir='trained',
                 mode='auto', period=1, debug=False):
        super(SaveCallback, self).__init__()
        self.model_to_save = model
        self.monitor_variable = monitor_variable
        self.verbose = verbose
        self.debug = debug
        # Check if directory exists
        ensure_folder(save_dir)
        _datetime = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        if self.debug:
            self.filepath = os.path.join(os.getcwd(), save_dir, 'model_trained.h5')
        else:
            self.filepath = os.path.join(os.getcwd(), save_dir,
                                         '%s-model_trained.{epoch:02d}-{%s:.2f}.h5' % (_datetime, monitor_variable))
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