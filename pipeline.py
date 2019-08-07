from keras.datasets import mnist
from preprocessing import ImageScaler, split_into_bags, add_color_channel
from utils import extend_bags_permutations
from sklearn.pipeline import Pipeline
from model import BagModel

MODEL_NAME = 'model_trained.h5'

pipeline = Pipeline([
   ('scaler', ImageScaler()),
   ('regressor', BagModel(save_to=MODEL_NAME, num_epochs=10))
])

pipeline_load = Pipeline([
    ('scaler', ImageScaler()),
    ('regressor', BagModel(load_from=MODEL_NAME))
])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=100,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.05)
    bags_x, bags_y = extend_bags_permutations(bags_x, bags_y, total_num=10)
    test_bags_x, test_bags_y = split_into_bags(x_test, y_test,
                                               bag_size=100,
                                               zero_bags_percent=0.5,
                                               zeros_in_bag_percentage=0.15)
    #pipeline_load.fit(test_bags_x, test_bags_y)
    #pipeline_load.score(test_bags_x, test_bags_y)
    # Model is saved automatically
    bags_x, test_bags_x = add_color_channel(bags_x), add_color_channel(test_bags_x)
    pipeline.fit(bags_x, bags_y)

    # Model loads
    pipeline_load.fit(bags_x, bags_y)
    # And score in calculated
    print('Score is: {}'.format(pipeline_load.score(test_bags_x, test_bags_y)))
