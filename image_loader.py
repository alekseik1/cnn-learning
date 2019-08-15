import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import extend_to_bagsize
from utils import split_into_bags, logger


def load_and_split_data(args):
    # TODO: better logging
    logger.info('started loading diseased images...')
    diseased_paths, diseased_imgs = load_images(args.diseased_dir)
    logger.info('diseased images are loaded')
    logger.info('started loading healthy images...')
    healthy_paths, healthy_imgs = load_images(args.healthy_dir)
    logger.info('healthy images are loaded')
    if len(diseased_imgs.shape) == 3:
        logger.info('adding color channels to diseased images...')
        from mnist.preprocessing import add_color_channel
        diseased_imgs = add_color_channel(diseased_imgs)
        logger.info('color channel added for diseased images')
    if len(healthy_imgs.shape) == 3:
        logger.info('adding color channels to diseased images...')
        from mnist.preprocessing import add_color_channel
        healthy_imgs = add_color_channel(healthy_imgs)
        logger.info('color channel added for healthy images')

    diseased_imgs = extend_to_bagsize(args.bag_size, diseased_imgs)
    healthy_imgs = extend_to_bagsize(args.bag_size, healthy_imgs)

    logger.info('splitting into bags...')
    diseased_bag_x = split_into_bags(diseased_imgs, args.bag_size)
    diseased_bag_y = np.zeros(len(diseased_bag_x))
    logger.info('split into bags: diseased')

    healthy_bag_x = split_into_bags(healthy_imgs, args.bag_size)
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
    logger.info('got images paths...')
    random.shuffle(images_paths)
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
