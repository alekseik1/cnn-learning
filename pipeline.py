from keras.datasets import mnist
from preprocessing import ImageScaler, split_into_bags
from sklearn.pipeline import Pipeline
from model import BagModel

MODEL_NAME = 'model_trained.h5'

pipeline = Pipeline([
   ('scaler', ImageScaler()),
   ('regressor', BagModel(save_to=MODEL_NAME))
])

pipeline_load = Pipeline([
    ('scaler', ImageScaler()),
    ('regressor', BagModel(load_from=MODEL_NAME))
])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=1000,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.15)
    pipeline.fit(bags_x, bags_y)
