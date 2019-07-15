NUM_EPOCHS = 60

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense
import numpy as np
from keras.datasets import mnist

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
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# Train it
autoencoder.fit(
    x_train, x_train, 
    epochs=NUM_EPOCHS, 
    batch_size=128, 
    shuffle=True, 
    validation_data=(x_test, x_test)
)

# Save the model
FILENAME = 'example_save.h5'
autoencoder.save(FILENAME)

