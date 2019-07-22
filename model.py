from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras import Model
from sklearn.base import BaseEstimator, ClassifierMixin
from preprocessing import preprocess_categories
from keras.models import load_model
import numpy as np


# TODO: make better name for the class
class BagModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 load_from=None,
                 optimizer='adadelta',
                 classifier_loss='categorical_crossentropy',
                 decoder_loss='binary_crossentropy',
                 classifier_metrics='accuracy',
                 num_epochs=10,
                 batch_size=128,
                 save_to=None):
        self.optimizer = optimizer
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        self.classifier_metrics = classifier_metrics
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_ = None
        self.save_to = save_to
        self.load_from = load_from

    def _create_model(self, input_shape):
        input_img = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        # After encoding, we need to classify images
        flatten = Flatten()(encoded)
        classifier = Dense(2, activation='softmax', name='classifier_output')(flatten)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        # In our model, each image in 'bag' should be considered as 'color channel' in CNN terms
        decoded = Conv2D(input_shape[-1], (3, 3), activation='relu', padding='same', name='decoded_output')(x)

        model = Model(inputs=[input_img], outputs=[classifier, decoded])
        model.compile(optimizer=self.optimizer,
                      # We define loss function for each output
                      loss={'classifier_output': self.classifier_loss, 'decoded_output': self.decoder_loss},
                      # And resulting loss function will be a weighted sum of all loss functions
                      # We want weigths 1.0 for all losses (for now, at least)
                      loss_weights={'classifier_output': 1.0, 'decoded_output': 1.0},
                      metrics={'classifier_output': self.classifier_metrics}
                      )
        return model

    def _ensure_model(self, input_shape):
        # Create model if necessary
        if self.load_from is not None:
            self.model_ = load_model(self.load_from)
        else:
            self.model_ = self._create_model(input_shape)

    def fit(self, x_train, y_train):
        # TODO: Validation of parameters

        # NOTE: we make category matrix from y_train here!
        y_train = preprocess_categories(y_train)

        self._ensure_model(input_shape=x_train.shape[1:])
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

    def save(self, path):
        self.model_.save(path)
