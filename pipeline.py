from utils import parse_args
from image_loader import KerasImageLoader
from config import load_config
import os


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config_type)

    from model import BagModel
    loader = KerasImageLoader(classes_directory='debug_imgs', bag_size=10, batch_size=2)

    model_instance = BagModel(num_epochs=config.epochs,

                              # TODO: rename to `load_weights`
                              load_path=(os.path.join(os.getcwd(), config.weights_dir, config.weights_name)
                                         if config.weights_name else None),
                              verbose=config.verbose,
                              batch_size=config.batch_size,
                              save_best_only=config.save_best_only, tensorboard_dir=config.tensorboard_dir,
                              debug=config.debug)
    model_instance.fit_generator(loader)
