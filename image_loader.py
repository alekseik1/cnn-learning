###################################################
# Here you'll find all code about loading images  #
# both from MNIST and folders.                    #
###################################################


import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import extend_to_bagsize
from utils import split_into_bags, logger, add_color_channel


def _separate_by_labels(x_data, y_data):
    return x_data[np.argwhere(y_data == 0).reshape(-1)], x_data[np.argwhere(y_data != 0).reshape(-1)]


def mnist_split_into_bags(x_data, y_data, zero_bags_percent=0.5, bag_size=100, zeros_in_bag_percentage=0.15):
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

    x_bags_split = np.empty([total_bags_number, bag_size, *x_data.shape[1:]])
    y_bags_split = np.empty([total_bags_number], dtype=np.int)

    # Create zero bags
    for i in range(zero_bags_number):
        x_bags_split[i], y_bags_split[i] = create_bag(zero_x, nonzero_x, bag_size, percentage=zeros_in_bag_percentage)
    # Create nonzero bags
    for i in range(zero_bags_number, total_bags_number):
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
    return result_x, result_y


def load_mnist(bag_size, zeros_in_bag, zero_bags):
    from keras.datasets import mnist
    # TODO: MNIST loader
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = add_color_channel(x_train), add_color_channel(x_test)
    bags_x, bags_y = mnist_split_into_bags(x_train, y_train,
                                           bag_size=bag_size,
                                           zero_bags_percent=zero_bags,
                                           zeros_in_bag_percentage=zeros_in_bag)
    test_bags_x, test_bags_y = mnist_split_into_bags(x_test, y_test,
                                                     bag_size=bag_size,
                                                     zero_bags_percent=zero_bags,
                                                     zeros_in_bag_percentage=zeros_in_bag)
    return (bags_x, bags_y), (test_bags_x, test_bags_y)


def load_and_split_data(diseased_dir, load_part, healthy_dir, bag_size):
    # TODO: better logging
    logger.info('started loading diseased images...')
    diseased_paths, diseased_imgs = load_images(diseased_dir, load_part=load_part)
    logger.info('diseased images are loaded')
    logger.info('started loading healthy images...')
    healthy_paths, healthy_imgs = load_images(healthy_dir, load_part=load_part)
    logger.info('healthy images are loaded')
    if len(diseased_imgs.shape) == 3:
        logger.info('adding color channels to diseased images...')
        from utils import add_color_channel
        diseased_imgs = add_color_channel(diseased_imgs)
        logger.info('color channel added for diseased images')
    if len(healthy_imgs.shape) == 3:
        logger.info('adding color channels to diseased images...')
        from utils import add_color_channel
        healthy_imgs = add_color_channel(healthy_imgs)
        logger.info('color channel added for healthy images')

    diseased_imgs = extend_to_bagsize(bag_size, diseased_imgs)
    healthy_imgs = extend_to_bagsize(bag_size, healthy_imgs)

    logger.info('splitting into bags...')
    diseased_bag_x = split_into_bags(diseased_imgs, bag_size)
    diseased_bag_y = np.zeros(len(diseased_bag_x))
    logger.info('split into bags: diseased')

    healthy_bag_x = split_into_bags(healthy_imgs, bag_size)
    healthy_bag_y = np.ones(len(healthy_bag_x))
    logger.info('split into bags: healthy')
    all_bags_x, all_bags_y = np.concatenate((diseased_bag_x, healthy_bag_x)), \
                             np.concatenate((diseased_bag_y, healthy_bag_y))

    logger.info('making train-test split...')
    train_bags_x, test_bags_x, train_bags_y, test_bags_y = train_test_split(all_bags_x, all_bags_y,
                                                                            test_size=0.33)
    logger.info('train-test split is ready')
    logger.info('train shape is {} | test shape is {}'.format(train_bags_x.shape, test_bags_x.shape))

    return (train_bags_x, train_bags_y), (test_bags_x, test_bags_y)


def load_images(filepath, mask='*.png', load_part=1.0):
    """
    Loads all images from folder

    :param filepath: folder, containing images. Non-image files
    :param mask: regexp for image search, e.g. '*.png'
    :param load_part: only only part of all images. 1.0 by default, meaning all images are loaded
    :return: (paths_list, ndarray_imgs) -- list of paths and numpy array of images
    """
    import random
    from skimage.io import imread
    images_paths = get_paths_list(filepath, mask)
    logger.info('got images paths...')
    random.shuffle(images_paths)
    images_paths = images_paths[0:int(len(images_paths)*load_part)]
    result = []
    i, N = 1, 100
    for path in images_paths:
        # Log every N images
        if i % N == 0:
            logger.info('loaded {}/{} images'.format(i, len(images_paths)))
        result.append(np.array(imread(path)))
        i += 1
    return images_paths, np.array(result)


def get_paths_list(filepath, mask):
    import glob
    import os
    path = os.path.join(filepath, mask)
    return glob.glob(path)
