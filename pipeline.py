from keras.datasets import mnist
from preprocessing import ImageScaler, split_into_bags, add_color_channel
from utils import extend_bags_permutations, parse_args
from sklearn.pipeline import Pipeline
from model import BagModel
import os

if __name__ == '__main__':
    args = parse_args()

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=args.epochs,
                               load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                          if args.load_from else None),
                               verbose=args.verbose,
                               save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                               debug=args.debug))
    ])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=100,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.05)
    bags_x, bags_y = extend_bags_permutations(bags_x, bags_y, total_num=10)
    test_bags_x, test_bags_y = split_into_bags(x_test, y_test,
                                               bag_size=100,
                                               zero_bags_percent=0.5,
                                               zeros_in_bag_percentage=0.05)
    bags_x, test_bags_x = add_color_channel(bags_x), add_color_channel(test_bags_x)
    pipeline.fit(bags_x, bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(bags_x, bags_y))
