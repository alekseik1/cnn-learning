from utils import parse_args
from image_loader import KerasImageLoader
from sklearn.pipeline import Pipeline
import os
import numpy as np
from skimage.io import imread
import itertools


if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler


    #loader = ImageLoader(class_paths=['debug_imgs/diseased', 'debug_imgs/healthy'],
    #                     batch_size=10, bag_size=50, class_weights=(0.5, 0.5))
    loader = KerasImageLoader(classes_directory='debug_imgs', bag_size=10, batch_size=2)

    model_instance = BagModel(num_epochs=args.epochs,

             # TODO: rename to `load_weights`
             load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                        if args.load_from else None),
             verbose=args.verbose,
             batch_size=args.batch_size,
             save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
             debug=args.debug)
    model = model_instance._create_model((loader.bag_size, *loader.image_shape))
    model.fit_generator(loader, steps_per_epoch=loader.total_images / args.batch_size, epochs=args.epochs)


    #(train_bags_x, train_bags_y), (test_bags_x, test_bags_y) = load_and_split_data(args)

"""
    pipeline = Pipeline([
        ('scaler', ImageScaler()),
        ('regressor', BagModel(num_epochs=args.epochs,
                               # TODO: rename to `load_weights`
                               load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                                          if args.load_from else None),
                               verbose=args.verbose,
                               batch_size=args.batch_size,
                               save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
                               debug=args.debug))
    ])
"""
#pipeline.fit(train_bags_x, train_bags_y)

# print('TEST: Score on test data: ', pipeline.score(test_bags_x, test_bags_y))
    #print('TRAIN: Score on test data: ', pipeline.score(train_bags_x, train_bags_y))
