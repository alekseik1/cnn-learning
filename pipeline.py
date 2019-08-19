from utils import parse_args
from image_loader import KerasImageLoader
import os


if __name__ == '__main__':
    args = parse_args()

    from model import BagModel
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
