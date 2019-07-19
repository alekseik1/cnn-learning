from keras.utils import to_categorical
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def create_bags(zero_x, nonzero_x, bag_size=100, percentage=0.01):
    zeros_in_bag = np.ceil(percentage*bag_size).astype(int)
    total_bags = np.ceil(
        (len(zero_x) + len(nonzero_x))/bag_size
    ).astype(int)
    result_x = np.empty(
        [total_bags, bag_size, *zero_x.shape[1:]]
    )
    result_y = np.empty([total_bags])
    for i in range(total_bags):
        # Numpy random.choice needs 1-D array
        zero_choice = zero_x[
            np.random.choice(len(zero_x), zeros_in_bag)
        ]
        # Fill rest data with nonzero_x
        nonzero_choice = nonzero_x[
            np.random.choice(len(nonzero_x), bag_size - zeros_in_bag)
        ]
        result_x[i] = np.concatenate((zero_choice, nonzero_choice))
        # If percentage is 0, we don't have zeros in a bag
        # so that the label is 1
        result_y[i] = int(percentage == 0)
    return result_x, result_y


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


ImageScaler = lambda: FunctionTransformer(preprocess_image, validate=False)
