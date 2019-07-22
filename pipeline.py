from keras.datasets import mnist
from preprocessing import ImageScaler, split_into_bags
from sklearn.pipeline import Pipeline
from model import BagModel

MODEL_NAME = 'model_trained.h5'
# TODO: include `bag_size` in model `fit()` method (so that no need to pass it to constructor)
BAG_SIZE = 6000

pipeline = Pipeline([
   ('scaler', ImageScaler()),
   ('regressor', BagModel(save_to=MODEL_NAME, bag_size=BAG_SIZE))
])

pipeline_load = Pipeline([
    ('scaler', ImageScaler()),
    ('regressor', BagModel(load_from=MODEL_NAME, bag_size=BAG_SIZE))
])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    bag_size = 6000

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=BAG_SIZE,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.15)

    # pipeline.fit(x_train, y_train)
    pipeline.fit(bags_x, bags_y)
