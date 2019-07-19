from keras.datasets import mnist
from preprocessing import ImageScaler, preprocess_categories, create_bags
from sklearn.pipeline import Pipeline
from model import MainModel
import numpy as np

MODEL_NAME = 'model_trained.h5'

if __name__ == '__main__':
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ##############################
    # Bag example
    x_train_zeros = x_train[np.argwhere(y_train == 0).reshape(-1)]
    x_test_zeros = x_test[np.argwhere(y_test == 0).reshape(-1)]
    x_train_nonzeros = x_train[np.argwhere(y_train != 0).reshape(-1)]
    x_test_nonzeros = x_test[np.argwhere(y_test != 0).reshape(-1)]
    ##############################

    x_bag, y_bag = create_bags(x_train_zeros, x_train_nonzeros)

    y_train, y_test = preprocess_categories(y_train), preprocess_categories(y_test)

    pipeline = Pipeline([
       ('scaler', ImageScaler()),
       ('regressor', MainModel(save_after_train=True))
    ])
    pipeline.fit(x_train, y_train)
