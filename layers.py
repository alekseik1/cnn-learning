from keras.layers import Lambda
import keras.backend as K
import numpy as np


def ImagesToBagLayer(bag_size):
    return Lambda(lambda all_images: K.reshape(all_images,
                                               (-1, bag_size, *all_images.shape[1:])))


def BagsToListLayer(bag_size):
    return Lambda(lambda all_bags: [all_bags[:, i] for i in range(bag_size)])


def _attach_to_pipeline(layer, pipeline):
    result = []
    # Connect other layers with each other
    for i, curr_layer in enumerate(pipeline):
        result.append(
            # Connect first layer to `layer`
            curr_layer(layer if i == 0 else result[i - 1])
        )
    return result[-1]