from utils import parse_args, load_and_split_data
from sklearn.pipeline import Pipeline
import os

if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler
    (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(args)

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=args.epochs,
                               load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                          if args.load_from else None),
                               verbose=args.verbose,
                               save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                               debug=args.debug))
    ])
    pipeline.fit(train_bags_x, train_bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(train_bags_x, train_bags_y))
