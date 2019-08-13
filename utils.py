import argparse
from itertools import islice, permutations
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Network to process images')
    parser.add_argument('--work_dir', type=str, default='trained',
                        help='directory to save trained model after epochs and load it from. '
                             'Will be created if not exist')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Whether to save only best (by accuracy) weights or everything')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--verbose', '-v', action='store_true', help='Be more verbose')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode. FOR NOW: affects only weights saver')
    parser.add_argument('--load_from', '-l', help='Filename of model weights to load')
    parser.add_argument('--tensorboard_dir', help='Directory to store tensorboard logs')
    return parser.parse_args()


def test_layer(layer, data):
    from keras import Model
    from keras.models import Input
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data)


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


def ensure_folder(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def load_mnist_bags(bag_size):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = extend_rotations(x_train, multiply_by=BAG_SIZE//10)

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=bag_size,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.05)
    test_bags_x, test_bags_y = split_into_bags(x_test, y_test,
                                               bag_size=bag_size,
                                               zero_bags_percent=0.5,
                                               zeros_in_bag_percentage=0.05)
    return (bags_x, bags_y), (test_bags_x, test_bags_y)


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = add_color_channel(x_train), add_color_channel(x_test)
    return (x_train, y_train), (x_test, y_test)