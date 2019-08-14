from sklearn.preprocessing import FunctionTransformer
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


def extend_rotations(data, multiply_by):
    result = data
    for i in range(multiply_by - 1):
        tmp = [rotate_image(img) for img in data]
        result = np.concatenate((result, np.array(tmp)))
        print('finished iteration {}'.format(i + 1))
    return result


def preprocess_image(x_data):
    x_data = x_data/np.max(x_data)
    return x_data


ImageScaler = lambda: FunctionTransformer(preprocess_image, validate=False)
