from utils import parse_args
from mnist.utils import load_mnist_bags
from sklearn.pipeline import Pipeline
import os

BAG_SIZE = 100

if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=args.epochs,
                               load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                          if args.load_from else None),
                               verbose=args.verbose,
                               save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                               debug=args.debug))
    ])

    bags_x, bags_y, test_bags_x, test_bags_y = load_mnist_bags(BAG_SIZE)
    pipeline.fit(bags_x, bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(bags_x, bags_y))
