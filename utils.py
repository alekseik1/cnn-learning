import argparse
import logging

# TODO: configure me based on verbosity level
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Network to process images')
    parser.add_argument('--config_type', '-c', help='type of config: "debug", "test" or "production"')

    args = parser.parse_args()
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
