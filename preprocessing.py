from keras.utils import to_categorical
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def _separate_by_labels(x_data, y_data):
    return x_data[np.argwhere(y_data == 0).reshape(-1)], x_data[np.argwhere(y_data != 0).reshape(-1)]


def split_into_bags(x_data, y_data, zero_bags_percent=0.5, bag_size=100, zeros_in_bag_percentage=0.15):
    """
    Splits data in bags so that it has `zero_bags_percent` bags with `bag_percentage` zero rate out of total bags
    and (1 - zero_bags_percent) of bags without any zero element

    :param x_data: source data
    :param y_data: source labels
    :param zero_bags_percent: percentage of bags containing at least one zero element
    :param bag_size: size of one bag
    :param zeros_in_bag_percentage: percentage of zero elements in one zero bag
    :return: tuple (x_bags_split, y_bags_split) where `x_bags` split is
    (bags_total, bag_size, img_width, img_height) array
    and `y_bags_split` is a 1-D vector (bags_total,) of numbers: 0 or 1
    """
    zero_x, nonzero_x = _separate_by_labels(x_data, y_data)
    total_bags_number = np.ceil(len(x_data)/bag_size).astype(int)
    zero_bags_number = np.ceil(total_bags_number*zero_bags_percent).astype(int)

    x_bags_split = np.empty([total_bags_number, *x_data.shape[1:], bag_size])
    y_bags_split = np.empty([total_bags_number], dtype=np.int)

    # Create zero bags
    for i in range(zero_bags_number):
        print('prepocessing zero-bag: {} of total {}'.format(i+1, total_bags_number))
        x_bags_split[i], y_bags_split[i] = create_bag(zero_x, nonzero_x, bag_size, percentage=zeros_in_bag_percentage)
    # Create nonzero bags
    for i in range(zero_bags_number, total_bags_number):
        print('preprocessing nonzero-bag: {} of total {}'.format(i+1, total_bags_number))
        x_bags_split[i], y_bags_split[i] = create_bag(zero_x, nonzero_x, bag_size, percentage=0)

    return x_bags_split, y_bags_split


def create_bag(zero_x, nonzero_x, bag_size=100, percentage=0.01):
    """
    For given `zero_x`, `nonzero_x` features, constructs one bag containing `bag_size*percentage` zeros.

    :param zero_x: source input (zero-labeled)
    :param nonzero_x: source input (nonzero-labeled)
    :param bag_size: size of a bag
    :param percentage: percentage of zeros
    :return: tuple (bag, label). `bag` is (bag_size, img_width, img_height) array, `label` is 1-D number -- one of 1 or 0
    """
    zeros_in_bag = np.ceil(percentage*bag_size).astype(int)
    # Numpy random.choice needs 1-D array
    zero_choice = zero_x[np.random.choice(len(zero_x), zeros_in_bag)]
    # Fill rest data with nonzero_x
    nonzero_choice = nonzero_x[np.random.choice(len(nonzero_x), bag_size - zeros_in_bag)]
    result_x = np.concatenate((zero_choice, nonzero_choice))
    result_y = int(percentage == 0)
    # Use .T for output shape (width, height, pic_amount)
    return result_x.T, result_y


def preprocess_categories(y_data):
    num_classes = len(np.unique(y_data))
    y_data = to_categorical(y_data, num_classes)
    return y_data


def preprocess_image(x_data):
    x_data = x_data/256
    return x_data


ImageScaler = lambda: FunctionTransformer(preprocess_image, validate=False)
