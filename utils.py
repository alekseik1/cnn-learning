import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from config import ProductionConfig, DebugConfig


def get_paths_list(filepath, mask):
    import glob
    import os
    path = os.path.join(filepath, mask)
    return glob.glob(path)


def load_images(filepath, mask='*.png'):
    """
    Loads all images from folder

    :param filepath: folder, containing images. Non-image files
    :param mask: regexp for image search, e.g. '*.png'
    :return: (paths_list, ndarray_imgs) -- list of paths and numpy array of images
    """
    import random
    from skimage.io import imread
    images_paths = get_paths_list(filepath, mask)
    random.shuffle(images_paths)
    result = []
    for path in images_paths:
        result.append(np.array(imread(path)))
    return images_paths, np.array(result)


def split_into_bags(array, bag_size):
    if array.shape[0] % bag_size != 0:
        raise ValueError("Length {} of array can't be by {}".format(len(array), bag_size))
    return np.reshape(array, (-1, bag_size, *array.shape[1:]))


def parse_args():
    parser = argparse.ArgumentParser(description='Network to process images')
    parser.add_argument('--config', action='store_true', help='load settings from config file')
    parser.add_argument('--work_dir', type=str, default='trained',
                        help='directory to save trained model after epochs and load it from. '
                             'Will be created if not exist')
    parser.add_argument('--save_best_only', action='store_true',
                        help='whether to save only best (by accuracy) weights or everything')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--verbose', '-v', action='store_true', help='be more verbose')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode. FOR NOW: affects only weights saver')
    parser.add_argument('--load_from', '-l', help='filename of model weights to load')
    parser.add_argument('--tensorboard_dir', '--tb_dir',
                        dest='tensorboard_dir', help='directory to store tensorboard logs')
    parser.add_argument('--bag_size', default='auto', help='size of a bag')
    parser.add_argument('--diseased_dir', help='path to diseased images')
    parser.add_argument('--healthy_dir', help='path to healthy images')
    args = parser.parse_args()
    if args.config:
        return DebugConfig if args.debug else ProductionConfig
    return args


def test_layer(layer, data):
    from keras import Model
    from keras.models import Input
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data)


def ensure_folder(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def get_best_bag_size(arr1, arr2):
    length = arr1.shape[0] + arr2.shape[0]
    for i in range(length-1, 0, -1):
        if length % i == 0:
            if i == 1:
                return length
            return i


def load_and_split_data(args):
    # TODO: better logging
    print('started loading data...')
    diseased_paths, diseased_imgs = load_images(args.diseased_dir)
    healthy_paths, healthy_imgs = load_images(args.healthy_dir)

    args.bag_size = (get_best_bag_size(diseased_imgs, healthy_imgs) if args.bag_size == 'auto'
                     else int(args.bag_size))

    print('splitting into bags...')
    diseased_bag_x = split_into_bags(diseased_imgs, args.bag_size)
    diseased_bag_y = np.zeros(len(diseased_bag_x))

    healthy_bag_x = split_into_bags(healthy_imgs, args.bag_size)
    healthy_bag_y = np.ones(len(healthy_bag_x))
    all_bags_x, all_bags_y = np.concatenate((diseased_bag_x, healthy_bag_x)), \
                             np.concatenate((diseased_bag_y, healthy_bag_y))

    print('making train-test-split...')
    train_bags_x, train_bags_y, test_bags_x, test_bags_y = train_test_split(all_bags_x, all_bags_y)

    return (train_bags_x, train_bags_y), (test_bags_x, test_bags_y)