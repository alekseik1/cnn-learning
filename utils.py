import argparse
import logging
import numpy as np
from config import ProductionConfig, DebugConfig

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
    parser.add_argument('--bag_size', type=int, default=50, help='size of a bag')
    parser.add_argument('--diseased_dir', help='path to diseased images')
    parser.add_argument('--healthy_dir', help='path to healthy images')
    parser.add_argument('--batch_size', type=int, default=128, help='size of one batch during training. 128 by default')
    parser.add_argument('--load_part', type=float, default=1.0, help='load only part of all images. '
                                                                     '1 by default, all images are loaded')
    args = parser.parse_args()
    if args.config:
        return DebugConfig if args.debug else ProductionConfig
    elif not args.diseased_dir or not args.healthy_dir:
        raise argparse.ArgumentError('You should either give --config or provide both --diseased_dir and --healthy_dir')
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


def get_best_bag_size(arr1: np.ndarray, arr2: np.ndarray):
    curr = min(arr1.shape[0], arr2.shape[0])//10
    while curr > 0:
        if arr1.shape[0] % curr == 0 and arr2.shape[0] % curr == 0:
            logger.info('Found best bag size: {}'.format(curr))
            return curr
        curr -= 1
