class Config:
    """
    Basic config
    """
    weights_dir = 'trained'
    save_best_only = True
    epochs = 10
    verbose = False
    debug = False
    load_part = 1.0
    weights_file = None
    tensorboard_dir = 'tlogs'
    bag_size = 50
    batch_size = 10
    diseased_dir = 'diseased_imgs'
    healthy_dir = 'healthy_imgs'


# For small learning checks
class DebugConfig(Config):
    debug = True
    verbose = True
    epochs = 10
    diseased_dir = 'debug_imgs/diseased'
    healthy_dir = 'debug_imgs/healthy'


# For small evaluation and model loading checks
class TestConfig(Config):
    verbose = True
    epochs = 10
    weights_file = 'model_trained.h5'
    diseased_dir = 'debug_imgs/diseased'
    healthy_dir = 'debug_imgs/healthy'


# For training on real data
class ProductionConfig(Config):
    epochs = 50
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/control/1'


def load_config(args):
    if args.config_type == 'debug':
        return DebugConfig
    elif args.config_type == 'production':
        return ProductionConfig
    elif args.config_type == 'test':
        return TestConfig
