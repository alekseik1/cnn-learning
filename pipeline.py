from utils import parse_args
from image_loader import load_and_split_data
from sklearn.pipeline import Pipeline
from config import load_config
import os

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    from model import BagModel
    from preprocessing import ImageScaler
    (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(config)

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=config.epochs,
                               model_weights_path=(os.path.join(os.getcwd(), config.weights_dir, config.weights_file)
                                          if config.weights_file else None),
                               verbose=config.verbose,
                               batch_size=config.batch_size,
                               save_best_only=config.save_best_only, tensorboard_dir=config.tensorboard_dir,
                               debug=config.debug))
    ])
    pipeline.fit(train_bags_x, train_bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(train_bags_x, train_bags_y))
