from itertools import islice, permutations

import numpy as np

from mnist.preprocessing import mnist_split_into_bags, add_color_channel
from preprocessing import extend_rotations


def extend_bags_permutations(x_bags, labels, total_num=100):
    """
    Shuffles elements in each bag, and then returns the dataset extended by permutations
    :param x_bags: original bags to permutate
    :param labels: labels corresponding to bags
    :param total_num: total numbers of permutations for each bag
    :return: (x_bags_new, y_labels_new) -- extended array of bags and corresponding labels
    """
    result_x, result_y = [], []
    for i in range(len(x_bags)):
        perms = list(islice(permutations(x_bags[i]), total_num))
        result_x.extend(perms)
        result_y.extend([labels[i] for _ in range(len(perms))])
    return np.array(result_x), np.array(result_y)


def load_mnist_bags(bag_size):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = extend_rotations(x_train, multiply_by=bag_size//10)

    bags_x, bags_y = mnist_split_into_bags(x_train, y_train,
                                           bag_size=bag_size,
                                           zero_bags_percent=0.5,
                                           zeros_in_bag_percentage=0.05)
    test_bags_x, test_bags_y = mnist_split_into_bags(x_test, y_test,
                                                     bag_size=bag_size,
                                                     zero_bags_percent=0.5,
                                                     zeros_in_bag_percentage=0.05)
    return (bags_x, bags_y), (test_bags_x, test_bags_y)


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = add_color_channel(x_train), add_color_channel(x_test)
    return (x_train, y_train), (x_test, y_test)