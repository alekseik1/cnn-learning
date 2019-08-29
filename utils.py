###################################################
# Here you'll find some utils for loading images, #
# reshaping arrays, parsing args and all other    #
# routine rubbish that need to be done before     #
# training the model.                             #
###################################################


import argparse
import logging
import numpy as np
from config import CONFIG_TYPES
from PIL import Image
import os

# TODO: configure me based on verbosity level
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def add_color_channel(x_data):
    return x_data.reshape(*x_data.shape, 1)


def split_into_bags(array, bag_size):
    if array.shape[0] % bag_size != 0:
        raise ValueError("Length {} of array can't be by {}".format(len(array), bag_size))
    return np.reshape(array, (-1, bag_size, *array.shape[1:]))


def parse_args():
    parser = argparse.ArgumentParser(description='Network to process images')
    parser.add_argument('--config_type', '-c', required=True,
                        help=f'config type. Available configs: {", ".join(CONFIG_TYPES.keys())}')
    parser.add_argument('--label', '-l', required=True, help='name of subfolder where weights and TF graphs are saved')

    args = parser.parse_args()
    return args


def ensure_folder(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def print_config(config):
    logger.info('##### Printing config: ####')
    parent_props = set(dir(config.__class__))
    own_props = set(dir(config)) - parent_props
    for prop in own_props:
        logger.info('{} = {}'.format(prop, getattr(config, prop)))


def save_array_as_images(array, path):
    ensure_folder(path)
    if array.shape[-1] == 1:
        array = array.reshape(array.shape[:-1])
    for i, single_image in enumerate(array):
        im = Image.fromarray(single_image)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(os.path.join(path, f'{i}.png'))
