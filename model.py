from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, concatenate, LSTM, Reshape
from keras import Model, Sequential
from keras.models import load_model
import keras.backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def _attach_to_pipeline(layer, pipeline):
    result = []
    # Connect other layers with each other
    for i, curr_layer in enumerate(pipeline):
        result.append(
            # Connect first layer to `layer`
            curr_layer(layer if i == 0 else result[i - 1])
        )
    return result[-1]


def SplitBagLayer(bag_size):
    return Lambda(lambda all_bags: [all_bags[:, i] for i in range(bag_size)])


# TODO: make better name for the class
class BagModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 load_from=None,
                 optimizer='adadelta',
                 classifier_loss='binary_crossentropy',
                 classifier_activation='sigmoid',
                 decoder_loss='binary_crossentropy',
                 classifier_metrics='accuracy',
                 num_epochs=10,
                 batch_size=128,
                 save_to=None):
        self.optimizer = optimizer
        self.classifier_loss = classifier_loss
        self.classifier_activation = classifier_activation
        self.decoder_loss = decoder_loss
        self.classifier_metrics = classifier_metrics
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_ = None
        self.save_to = save_to
        self.load_from = load_from

    def _create_model(self, input_shape):
        input_img = Input(shape=input_shape)

        # Create shared encoder.
        encoder_pipeline = [
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
        ]

        # Split bag into single images to get encoded vectors
        splitted_imgs = SplitBagLayer(bag_size=input_shape[0])(input_img)
        encoded_img_matrices, encoded_img_vectors = [], []
        for single_image in splitted_imgs:
            encoded_img = _attach_to_pipeline(single_image, encoder_pipeline)

            encoded_img_matrices.append(encoded_img)
            encoded_img_vectors.append(
                # We restore back `bag` dimension in vectors,
                # so that we'll have list of (batch_dim, 1, prod_dim) tensors
                Reshape((1, -1))(Flatten()(encoded_img))
            )
        # We have v=(vec1, ... , vecN) where N is number of images in one bag
        # Now we need to do aggregation
        concat_matrix = concatenate(encoded_img_vectors, axis=1)
        # Now we have array with shape (num_vectors, latent_features). Let's aggregate them
        # NOTE: Aggregator is based on maximum
        aggregator = Lambda(lambda matrix: K.max(matrix, axis=1))(concat_matrix)

        # After encoding, we need to classify images
        classifier = Dense(1, activation=self.classifier_activation, name='classifier_output')(aggregator)

        decoder_pipeline = [
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(input_shape[-1], (3, 3), activation='relu', padding='same'),
            # For convenience, reshape (None, w, h, c) -> (None, 1, w, h, c)
            # where 'w'=width, 'h'=height, 'c'=color_channel
            Reshape((1, *input_shape[1:]))
        ]
        # TODO: decoding
        decoded_images = [_attach_to_pipeline(single_image, decoder_pipeline) for single_image in encoded_img_matrices]
        decoded_images = concatenate(decoded_images, axis=1, name='decoded_output')

        model = Model(inputs=[input_img], outputs=[classifier, decoded_images])
        model.compile(optimizer=self.optimizer,
                      # We define loss function for each output
                      loss={'classifier_output': self.classifier_loss, 'decoded_output': self.decoder_loss},
                      # And resulting loss function will be a weighted sum of all loss functions
                      # We want weigths 1.0 for all losses (for now, at least)
                      loss_weights={'classifier_output': 1.0, 'decoded_output': 1.0},
                      metrics={'classifier_output': self.classifier_metrics}
                      )
        return model

    def fit(self, x_train, y_train):
        # TODO: Validation of parameters

        # NOTE: we make category matrix from y_train here!
        y_train = (y_train > 0).astype(int)

        self.model_ = self._create_model(x_train.shape[1:])
        if self.load_from:
            self.model_ = load_model(self.load_from)
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
                # validation_data=(x_test, {'classifier_output': y_test, 'decoded_output': x_test}),
            )
        if self.save_to:
            self.model_.save(self.save_to)
        return self

    def predict(self, x_data):
        return np.argmax(self.predict_proba(x_data), axis=1)

    def predict_proba(self, x_data):
        # NOTE. We do not return decoded pictures for two reasons:
        # 1. sklearn expect `predict` method to return one value
        # 2. We actually don't need decoded images
        classes, decoded_imgs = self.model_.predict(x_data)
        return classes
