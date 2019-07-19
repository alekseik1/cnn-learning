from keras.datasets import mnist
from preprocessing import ImageScaler, preprocess_categories, create_bags
from sklearn.pipeline import Pipeline
from model import MainModel
import numpy as np

MODEL_NAME = 'model_trained.h5'

pipeline = Pipeline([
   ('scaler', ImageScaler()),
   ('regressor', MainModel(save_to=MODEL_NAME))
])

pipeline_load = Pipeline([
    ('scaler', ImageScaler()),
    ('regressor', MainModel(load_from=MODEL_NAME))
])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train, y_test = preprocess_categories(y_train), preprocess_categories(y_test)

    ##############################
    # Bag example
    # x_train_zeros = x_train[np.argwhere(y_train == 0).reshape(-1)]
    # x_test_zeros = x_test[np.argwhere(y_test == 0).reshape(-1)]
    # x_train_nonzeros = x_train[np.argwhere(y_train != 0).reshape(-1)]
    # x_test_nonzeros = x_test[np.argwhere(y_test != 0).reshape(-1)]
    ##############################

    # x_bag, y_bag = create_bags(x_train_zeros, x_train_nonzeros)

    pipeline.fit(x_train, y_train)
