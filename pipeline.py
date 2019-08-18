from utils import parse_args
from image_loader import load_and_split_data
from sklearn.pipeline import Pipeline
import os


def data_generator(x_data, y_data):
    i = 0
    while i < len(x_data) - 2:
        i += 1
        yield (x_data[i:i+2], {'decoded_output': x_data[i:i+2], 'classifier_output': y_data[i:i+2]})

if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler
    (train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(args)
    train_bags_x = train_bags_x / 255.
    test_bags_x = test_bags_x / 255.

    model_instance = BagModel(num_epochs=args.epochs,
                           # TODO: rename to `load_weights`
                           load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                      if args.load_from else None),
                           verbose=args.verbose,
                           batch_size=args.batch_size,
                           save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                           debug=args.debug)
    model = model_instance._create_model(train_bags_x.shape[1:])
    model.fit_generator(data_generator(train_bags_x, train_bags_y), steps_per_epoch=len(train_bags_x)/2,
                        epochs=10)
    print()
