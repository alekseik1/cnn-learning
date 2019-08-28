from utils import parse_args
from image_loader import load_and_split_data, load_mnist
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from config import load_config, MNIST_config

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    if hasattr(config, 'zero_bags'):
        # If MNIST config is passed
        (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_mnist(
            bag_size=config.bag_size,
            zeros_in_bag=config.zeros_in_bag,
            zero_bags=config.zero_bags
        )
    else:
        (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(
            diseased_dir=config.diseased_dir,
            healthy_dir=config.healthy_dir,
            load_part=config.load_part,
            bag_size=config.bag_size
        )

    from model import BagModel
    from preprocessing import ImageScaler

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
                               classifier_loss_weight=1.0,
                               decoder_loss_weight=1.0,
                               num_epochs=config.epochs,
                               batch_size=config.batch_size,
                               verbose=config.verbose,
                               save_best_only=config.save_best_only,
                               debug=config.debug))
    ])

    # TODO: customize for your own GridSearch
    # We will cover many proportions
    param_grid = {
        'regressor__classifier_loss_weight': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        'regressor__decoder_loss_weight': [1.0],
    }
    clf = GridSearchCV(pipeline, param_grid, n_jobs=1, pre_dispatch=1)
    clf.fit(train_bags_x, train_bags_y)
    print(f'CV results are:')
    print(clf.cv_results_)
    print('----- Best estimator is:')
    print(clf.best_estimator_)
    #pipeline.fit(train_bags_x, train_bags_y)

    #print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    #print('TRAIN: Score on test data: ', pipeline.score(train_bags_x, train_bags_y))
