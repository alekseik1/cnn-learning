from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, \
    Flatten, Dense, Lambda, concatenate, Reshape, Dropout
from keras import Model
import os
from keras.models import load_model
from keras.callbacks import TensorBoard
import keras.backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from utils import ensure_folder, save_array_as_images
from callbacks import SaveCallback
import numpy as np
from layers import SplitBagLayer, _attach_to_pipeline

WEIGHTS_DIRECTORY = 'weights'
TENSORBOARD_DIRECTORY = 'tensorboard-logs'
# TODO: maybe move it to config?
# TODO: HARDCODED link. It can be broken or unaccessable
IMAGE_DIR = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/akozharin/images'


# TODO: make better name for the class
class BagModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 load_weights_from_file=None,
                 optimizer='adadelta',
                 label='unlabeled',
                 classifier_loss='binary_crossentropy',
                 classifier_activation='sigmoid',
                 decoder_loss='binary_crossentropy',
                 classifier_metrics='accuracy',
                 classifier_loss_weight=1.0,
                 decoder_loss_weight=1.0,
                 num_epochs=10,
                 batch_size=128,
                 verbose=False,
                 save_best_only=True,
                 debug=False):
        self.optimizer = optimizer
        self.label = label
        self.classifier_loss = classifier_loss
        self.classifier_activation = classifier_activation
        self.decoder_loss = decoder_loss
        self.classifier_metrics = classifier_metrics
        self.classifier_loss_weight = classifier_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_ = None
        self.load_weights_from_file = load_weights_from_file
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.debug = debug

    def _create_model(self, input_shape):
        input_img = Input(shape=input_shape)

        # Create shared encoder.
        encoder_pipeline = [
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same', strides=2),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same', strides=2),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
        ]

        # Split bag into single images to get encoded vectors
        splitted_imgs = SplitBagLayer(bag_size=input_shape[0])(input_img)
        encoded_img_matrices = []
        for single_image in splitted_imgs:
            encoded_img = _attach_to_pipeline(single_image, encoder_pipeline)

            encoded_img_matrices.append(encoded_img)
        # We have v=(vec1, ... , vecN) where N is number of images in one bag
        # Now we need to do aggregation
        concat_matrix = concatenate(
            [Reshape((1, -1))(
                Flatten()(img)
            ) for img in encoded_img_matrices],
            axis=1)
        # Now we have array with shape (num_vectors, latent_features). Let's aggregate them
        # NOTE: Aggregator is based on maximum

        # THIS IS THE PART WHERE WE LOOSE 1 DIMENSION (dimension of bags)
        aggregator = Lambda(lambda matrix: K.max(matrix, axis=1))(concat_matrix)

        # After encoding, we need to classify images
        classifier = Dense(128, activation=self.classifier_activation)(aggregator)
        classifier = Dropout(rate=0.5)(classifier)
        classifier = Dense(1, activation=self.classifier_activation, name='classifier_output')(classifier)

        decoder_pipeline = [
            # TODO: maybe make activation functions tunable?
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(input_shape[-1], (3, 3), activation='relu', padding='same'),
            # reshape (None, w, h, c) -> (None, 1, w, h, c) where 'w'=width, 'h'=height, 'c'=color_channel
            Reshape((1, *input_shape[1:]))
        ]
        decoded_images = [_attach_to_pipeline(single_image, decoder_pipeline) for single_image in encoded_img_matrices]
        decoded_images = concatenate(decoded_images, axis=1, name='decoded_output')

        model = Model(inputs=[input_img], outputs=[classifier, decoded_images])
        model.compile(optimizer=self.optimizer,
                      loss={'classifier_output': self.classifier_loss, 'decoded_output': self.decoder_loss},
                      loss_weights={'classifier_output': self.classifier_loss_weight,
                                    'decoded_output': self.decoder_loss_weight},
                      metrics={'classifier_output': self.classifier_metrics}
                      )
        return model

    def fit(self, x_train, y_train):
        # TODO: Validation of parameters
        # Train/validation split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          # TODO: mb make `test_size` tunable?
                                                          test_size=0.4, random_state=42)

        # NOTE: we make category matrix from y_train here!
        y_train = (y_train > 0).astype(int)

        self.model_ = self._create_model(x_train.shape[1:])

        weights_folder = os.path.join(os.getcwd(), self.label, WEIGHTS_DIRECTORY)
        # TODO: hardcoded monitor variable. Move it to config file
        callbacks = [SaveCallback(monitor_variable='val_classifier_output_acc',
                                  save_dir=weights_folder,
                                  model=self.model_,
                                  verbose=self.verbose,
                                  save_best_only=self.save_best_only,
                                  debug=self.debug)]

        # Take care of tensorboard
        tb_folder = os.path.join(os.getcwd(), self.label, TENSORBOARD_DIRECTORY)
        ensure_folder(tb_folder)
        callbacks.append(TensorBoard(log_dir=tb_folder))

        if self.load_weights_from_file:
            self.model_ = load_model(self.load_weights_from_file)
        else:
            # Train it
            self.model_.fit(
                # Train data
                x_train,
                # Test data. Note that each output has its own data to train on!
                {'decoded_output': x_train, 'classifier_output': y_train},
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(x_val, {'classifier_output': y_val, 'decoded_output': x_val}),
                callbacks=callbacks
            )
        return self

    def predict(self, x_data):
        return np.round(self.predict_proba(x_data).reshape(-1))

    def predict_proba(self, x_data):
        # NOTE. We do not return decoded pictures for two reasons:
        # 1. sklearn expect `predict` method to return one value
        # 2. We actually don't need decoded images
        # NOTE: uncomment these two pieces if you want decoded and original pictures to be saved on NAS
        '''
        # TODO: better post-processing of image (mb create some reverse function to pre-processing)
        save_array_as_images((255.*x_data).reshape(-1, *x_data.shape[2:]),
                             os.path.join(os.getcwd(), IMAGE_DIR, 'original'))
        '''
        classes, decoded_imgs = self.model_.predict(x_data)
        '''
        # TODO: better post-processing of image (mb create some reverse function to pre-processing)
        save_array_as_images((255.*x_data).reshape(-1, *x_data.shape[2:]),
        save_array_as_images((255.*decoded_imgs).reshape(-1, *decoded_imgs.shape[2:]),
                             os.path.join(os.getcwd(), IMAGE_DIR, 'decoded'))
        '''
        return classes
