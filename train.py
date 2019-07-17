from keras import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense
import numpy as np
from keras.datasets import mnist

from config import ENCODER_MODEL_NAME, NUM_EPOCHS

# load dataset
# MNIST, for now
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_width, img_height, img_depth = x_train.shape[1], x_train.shape[2], 1

# Data preparation
x_train = x_train/256
x_test = x_test/256
x_train = x_train.reshape(len(x_train), img_width, img_height, img_depth)
x_test = x_test.reshape(len(x_test), img_width, img_height, img_depth)

# Building model
input_img = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# After encoding, we need to classify images
classifier = Dense(784, activation='relu', name='classifier_output')(encoded)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same', name='decoded_output')(x)

model = Model(inputs=[input_img], outputs=[classifier, decoded])
model.compile(optimizer='adadelta',
              # We define loss function for each output
              loss={'classifier_output': 'binary_crossentropy', 'decoded_output': 'binary_crossentropy'},
              # And resulting loss function will be a weighted sum of all loss functions
              # We want weigths 1.0 for all losses (for now, at least)
              loss_weights={'classifier_output': 1.0, 'decoded_output': 1.0},
              )
# Train it
model.fit(
    # Train data
    x_train,
    # Test data. Note that each output has its own data to train on!
    {'decoded_output': x_train, 'classifier_output': y_train},
    epochs=NUM_EPOCHS, 
    batch_size=128, 
    shuffle=True, 
    validation_data=(x_test, x_test)
)

# Save the model
model.save(ENCODER_MODEL_NAME)

