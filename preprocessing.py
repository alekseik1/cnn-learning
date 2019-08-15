from sklearn.preprocessing import FunctionTransformer
from utils import logger
import numpy as np


def rotate_image(img):
    n_rot = np.random.choice([1, 2, 3, 4])
    if n_rot == 1:
        img = np.rot90(img, k=1, axes=(0, 1))
    elif n_rot == 2:
        img = np.rot90(img, k=2, axes=(0, 1))
    elif n_rot == 3:
        img = np.rot90(img, k=3, axes=(0, 1))
    return img


def extend_to_bagsize(bagsize: int, raw_images: np.ndarray):
    """
    Extends array of imgs by rotating its first elements so that array's length will be divisible by `bagsize`.
    Note that it can only multiply images by 8 (4 rotations + 1 flip)
    :param bagsize:
    :param raw_images:
    :raise: ValueError if it is impossible to extend array to fit `bagsize`
    :return: extended array
    """
    rotations_num = bagsize - (len(raw_images) % bagsize)
    if rotations_num > 8*len(raw_images):
        raise ValueError('Cannot extend images by only rotating and flipping. Bag size is too big')
    logger.info('need to add {} images'.format(rotations_num))
    for i in range(rotations_num):
        raw_images = np.concatenate((
            raw_images,
            rotate_image(raw_images[i]).reshape(1, *raw_images.shape[1:])
        ))
    return raw_images


def preprocess_image(x_data):
    x_data = x_data/np.max(x_data)
    return x_data


ImageScaler = lambda: FunctionTransformer(preprocess_image, validate=False)
