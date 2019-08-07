from itertools import islice, permutations

import numpy as np
from keras.models import Input
from keras import Model


def test_layer(layer, data):
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data)


def extend_bags_permutations(x_bags, labels, total_num=100):
    """
    Shuffles elements in each bag, and then returns the dataset extended by permutations
    :param x_bags: original bags to permutate
    :param labels: labels corresponding to bags
    :param total_num: total numbers of permutations for each bag
    :return: (x_bags_new, y_labels_new) -- extended array of bags and corresponding labels
    """
    result_x, result_y = [], []
    for i in range(len(x_bags)):
        perms = list(islice(permutations(x_bags[i]), total_num))
        result_x.extend(perms)
        result_y.extend([labels[i] for _ in range(len(perms))])
    return np.array(result_x), np.array(result_y)