from utils import parse_args
from image_loader import load_and_split_data
from sklearn.pipeline import Pipeline
from config import load_config

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    from model import BagModel
    from preprocessing import ImageScaler
    (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(config)

    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(load_weights_from_file=(config.weights_file if config.weights_file else None),
                               # TODO: read hardcoded options from config
                               optimizer='adadelta',
                               label=args.label,
                               classifier_loss='binary_crossentropy',
                               classifier_activation='sigmoid',
                               decoder_loss='binary_crossentropy',
                               classifier_metrics='accuracy',
                               num_epochs=config.epochs,
                               batch_size=config.batch_size,
                               verbose=config.verbose,
                               save_best_only=config.save_best_only,
                               debug=config.debug))
    ])
    pipeline.fit(train_bags_x, train_bags_y)

    print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    print('TRAIN: Score on test data: ', pipeline.score(train_bags_x, train_bags_y))
