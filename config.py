class Config:
    """
    Basic config
    """
    weights_dir = 'trained'
    save_best_only = True
    epochs = 10
    verbose = False
    debug = False
    weights_name = None
    tensorboard_dir = 'tlogs'
    bag_size = 50
    batch_size = 10


class DebugConfig(Config):
    debug = True
    verbose = True
    epochs = 10


class TestConfig(Config):
    debug = True
    verbose = True
    epochs = 10
    weights_name = 'model_trained.h5'


class ProductionConfig(Config):
    epochs = 50
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/control/1'


def load_config(config_type):
    if config_type == 'debug':
        return DebugConfig
    elif config_type == 'production':
        return ProductionConfig
    elif config_type == 'test':
        return TestConfig
