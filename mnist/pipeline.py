from utils import parse_args
from sklearn.pipeline import Pipeline
import os

BAG_SIZE = 100

if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler, split_into_bags, extend_rotations
    from mnist.preprocessing import add_color_channel

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=args.epochs,
                               load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                          if args.load_from else None),
                               verbose=args.verbose,
                               save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                               debug=args.debug))
    ])

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = add_color_channel(x_train), add_color_channel(x_test)
    x_train = extend_rotations(x_train, multiply_by=BAG_SIZE//10)

    bags_x, bags_y = split_into_bags(x_train, y_train,
                                     bag_size=BAG_SIZE,
                                     zero_bags_percent=0.5,
                                     zeros_in_bag_percentage=0.05)
    test_bags_x, test_bags_y = split_into_bags(x_test, y_test,
                                               bag_size=BAG_SIZE,
                                               zero_bags_percent=0.5,
                                               zeros_in_bag_percentage=0.05)
    pipeline.fit(bags_x, bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(bags_x, bags_y))
