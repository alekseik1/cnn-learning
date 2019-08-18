import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import extend_to_bagsize
from utils import split_into_bags, logger
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.io import imread
import glob
import os


class FolderLoader(object):

    def __init__(self, path: str, label, mask: str = '*.png'):
        self.path = path
        self.label = label
        self.found_files = glob.glob(os.path.join(path, mask))
        # Shuffle it!
        np.random.shuffle(self.found_files)
        self.backup_list = list(self.found_files)
        self.total_images = len(self.found_files)

    def get_image_shape(self):
        return imread(self.found_files[0]).shape

    def __next__(self):
        return self.next()

    def next(self):
        if len(self.found_files) == 0:
            self.found_files = list(self.backup_list)
            np.random.shuffle(self.found_files)
        return imread(self.found_files.pop())


class ImageLoader:

    def __init__(self, class_paths: list, batch_size: int, bag_size: int, class_weights, mask='*.png'):
        """
        On demand image loader
        """
        self.mask = mask
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.class_weights = np.array(class_weights)
        assert len(class_weights) == len(class_paths)
        self.loaders = []
        for number, path in enumerate(class_paths):
            self.loaders.append(
                FolderLoader(path=path, label=number, mask=mask))
        # NOTE: we read 1 file only to determine image shape
        self.image_shape = self.loaders[0].get_image_shape()
        self.total_images = sum([loader.total_images for loader in self.loaders])

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        # TODO: problem with cycled file read
        final_batch, final_labels = [], []
        bags_per_class = np.array(np.floor(self.batch_size*self.class_weights), dtype=int)
        # Ensure that bag size is not broken due to rounding errors
        while sum(bags_per_class) < self.batch_size:
            bags_per_class[-1] += 1
        for class_number, total_bags in enumerate(bags_per_class):
            for bag_number in range(total_bags):
                one_bag = np.empty((self.bag_size, *self.image_shape))
                for image_number in range(self.bag_size):
                    tmp = next(self.loaders[class_number])
                    one_bag[image_number] = tmp
                final_batch.append(one_bag)
                final_labels.append(class_number)
        final_batch, final_labels = np.array(final_batch), np.array(final_labels)
        final_batch = (final_batch if final_batch.shape == 5 else self.add_color_channel(final_batch))
        return (final_batch, {'decoded_output': final_batch, 'classifier_output': final_labels})


    """
        # AAA
        for i in range(self.batch_size):
            one_bag = np.empty((self.bag_size, *self.image_shape))
            for j in range(self.bag_size):
                tmp = imread(self.found_files.pop())
                if len(tmp.shape) == 2:
                    tmp = self.add_color_channel(tmp)
                one_bag[j] = tmp
            batch_x[i] = one_bag
        return batch_x
    """

    @staticmethod
    def add_color_channel(array):
        return np.reshape(array, (*array.shape, 1))

    @staticmethod
    def _get_paths(path, mask):
        mask = os.path.join(path, mask)
        result = glob.glob(mask)
        np.random.shuffle(result)
        return result

    """
    healthy_paths, healthy_imgs = load_images(args.healthy_dir, load_part=args.load_part)
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
    """

'''
def load_images(filepath, mask='*.png', load_part=1.0):
    """
    Loads all images from folder

    :param filepath: folder, containing images. Non-image files
    :param mask: regexp for image search, e.g. '*.png'
    :param load_part: only only part of all images. 1.0 by default, meaning all images are loaded
    :return: (paths_list, ndarray_imgs) -- list of paths and numpy array of images
    """
    import random
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
'''


"""
def get_paths_list(filepath, mask):
    logger.info(f'getting files for path {filepath} with mask "{mask}"')
    path = os.path.join(filepath, mask)
    result = glob.glob(path)
    logger.info(f'got files list for path {filepath}. Total files: {len(result)}')
    return result
"""
