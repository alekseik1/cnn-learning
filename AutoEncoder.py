#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().system('jupyter nbconvert --to=python AutoEncoder.ipynb')


# In[1]:


from keras import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


# In[2]:


# load dataset
# MNIST, for now
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_width, img_height, img_depth = x_train.shape[1], x_train.shape[2], 1


# In[3]:


# Data preparation
x_train = x_train/256
x_test = x_test/256
x_train = x_train.reshape(len(x_train), img_width, img_height, img_depth)
x_test = x_test.reshape(len(x_test), img_width, img_height, img_depth)


# In[4]:


input_img = Input(
    shape=(x_train.shape[1], x_train.shape[2], 1)
)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)


# In[5]:


autoencoder = Model(input_img, decoded)


# In[7]:


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[8]:


NUM_EPOCHS = 10
autoencoder.fit(
    x_train, x_train, 
    epochs=NUM_EPOCHS, 
    batch_size=128, 
    shuffle=True, 
    validation_data=(x_test, x_test)
)


# ## Save the model

# In[12]:


FILENAME = 'example_save.h5'
autoencoder.save(FILENAME)

