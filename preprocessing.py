from keras.utils import to_categorical
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def preprocess_categories(y_data):
    num_classes = len(np.unique(y_data))
    y_data = to_categorical(y_data, num_classes)
    return y_data


def preprocess_image(x_data):
    img_width, img_height = x_data.shape[1], x_data.shape[2]
    # If no depth is specified, consider it as 1
    img_depth = 1 if len(x_data.shape) == 3 else x_data.shape[3]

    x_data = x_data/256
    x_data = x_data.reshape(len(x_data), img_width, img_height, img_depth)
    return x_data


ImageScaler = FunctionTransformer(preprocess_image, validate=False)
