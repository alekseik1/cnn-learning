from utils import parse_args
from image_loader import ImageLoader
from sklearn.pipeline import Pipeline
import os
import numpy as np
from skimage.io import imread
import itertools


def bag_generator(batches, bag_size, image_shape):
    ############################################################
    # Here we just transform one batch to one bag. In future,
    # however, DataGen batch_size != bag_size so that we need
    # to reshape it in proper way.
    ############################################################
    print()
    batch_x, batch_y = next(batches)
    bag_x = batch_x.reshape((-1, bag_size, *image_shape))
    bag_y = np.array(1)
    yield np.zeros(2)


BATCH_SIZE = 10
def bag_batch_generator(ids, folder, batch_size=BATCH_SIZE):
    batch = []
    while True:
        np.random.shuffle(ids)
        for i in ids:
            batch.append(i)
            if len(batch) == batch_size:
                yield load_data(batch, folder)
                batch = []


def load_data(ids):
    X, y = [], []
    for i in ids:
        x = imread(f'{i}.png')
        y = 0





if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
    from preprocessing import ImageScaler


    loader = ImageLoader(class_paths=['debug_imgs/diseased', 'debug_imgs/healthy'],
                         batch_size=10, bag_size=50, class_weights=(0.5, 0.5))

    a = next(loader)
    model_instance = BagModel(num_epochs=args.epochs,

             # TODO: rename to `load_weights`
             load_path=(os.path.join(os.getcwd(), args.work_dir, args.load_from)
                        if args.load_from else None),
             verbose=args.verbose,
             batch_size=args.batch_size,
             save_best_only=args.save_best_only, tensorboard_dir=args.tensorboard_dir,
             debug=args.debug)
    model = model_instance._create_model((loader.bag_size, *loader.image_shape, 1))
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
