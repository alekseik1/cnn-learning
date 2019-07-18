from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras.datasets import mnist
from keras import Model
from keras.utils import to_categorical
from config import ENCODER_MODEL_NAME, NUM_CLASSES
from sklearn.base import BaseEstimator, ClassifierMixin


def preprocess_data(x_data, y_data, img_width, img_height, img_depth):
    y_data = to_categorical(y_data, NUM_CLASSES)

    x_data = x_data/256
    x_data = x_data.reshape(len(x_data), img_width, img_height, img_depth)
    return x_data, y_data


# TODO: make better name for the class
class MainModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 optimizer='adadelta',
                 input_shape=(28, 28, 1),
                 classifier_loss='categorical_crossentropy',
                 decoder_loss='binary_crossentropy',
                 classifier_metrics='accuracy',
                 num_epochs=10,
                 batch_size=128):
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        self.classifier_metrics = classifier_metrics
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_ = None

    def _create_model(self):
        input_img = Input(shape=self.input_shape)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        # After encoding, we need to classify images
        flatten = Flatten()(encoded)
        classifier = Dense(10, activation='softmax', name='classifier_output')(flatten)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='relu', padding='same', name='decoded_output')(x)

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

    def fit(self, x_train, y_train):
        # TODO: Validation of parameters

        # Create model if necessary
        if self.model_ is None:
            self.model_ = self._create_model()
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
        return self

    def predict(self, x_data):
        # NOTE. We do not return decoded pictures for two reasons:
        # 1. sklearn expect `predict` method to return one value
        # 2. We actually don't need decoded images
        classes, decoded_imgs = self._model.predict(x_data)
        return classes

    def save(self, path):
        self.model_.save(path)


"""
def create_model(input_shape, optimizer='adadelta'):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # After encoding, we need to classify images
    flatten = Flatten()(encoded)
    classifier = Dense(10, activation='softmax', name='classifier_output')(flatten)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same', name='decoded_output')(x)

    model = Model(inputs=[input_img], outputs=[classifier, decoded])
    model.compile(optimizer=optimizer,
                  # We define loss function for each output
                  loss={'classifier_output': 'categorical_crossentropy', 'decoded_output': 'binary_crossentropy'},
                  # And resulting loss function will be a weighted sum of all loss functions
                  # We want weigths 1.0 for all losses (for now, at least)
                  loss_weights={'classifier_output': 1.0, 'decoded_output': 1.0},
                  metrics={'classifier_output': 'accuracy'}
                  )
    return model
"""


if __name__ == '__main__':
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_width, img_height = x_train.shape[1], x_train.shape[2]
    img_depth = 1
    # Preprocessing
    x_train, y_train = preprocess_data(x_train, y_train, img_width, img_height, img_depth)
    x_test, y_test = preprocess_data(x_test, y_test, img_width, img_height, img_depth)
    """
    # Define model
    model = create_model((img_width, img_height, img_depth))
    # Train it!
    model.fit(
        # Train data
        x_train,
        # Test data. Note that each output has its own data to train on!
        {'decoded_output': x_train, 'classifier_output': y_train},
        epochs=NUM_EPOCHS,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, {'classifier_output': y_test, 'decoded_output': x_test}),
    )
    """
    model = MainModel()
    model.fit(x_train, y_train)
    model.save(ENCODER_MODEL_NAME)
