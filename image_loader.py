import numpy as np
from sklearn.model_selection import train_test_split

from utils import get_best_bag_size, split_into_bags


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
    train_bags_x, test_bags_x, train_bags_y, test_bags_y = train_test_split(all_bags_x, all_bags_y,
                                                                            test_size=0.33)

    return (train_bags_x, train_bags_y), (test_bags_x, test_bags_y)


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


def get_paths_list(filepath, mask):
    import glob
    import os
    path = os.path.join(filepath, mask)
    return glob.glob(path)