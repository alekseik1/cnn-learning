from keras.datasets import mnist
from preprocessing import ImageScaler, preprocess_categories, split_into_bags
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

    bags = split_into_bags(x_train, y_train, bag_size=6000)

    pipeline.fit(x_train, y_train)
