class Config:
    """
    Basic config
    """
    weights_file = None
    optimizer = 'adadelta'
    classifier_loss = 'binary_crossentropy'
    classifier_activation = 'sigmoid'
    decoder_loss = 'binary_crossentropy'
    classifier_metrics = 'accuracy'
    epochs = 10
    batch_size = 10
    verbose = False
    save_best_only = True
    debug = False
    load_part = 1.0
    bag_size = 50
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


class ProductionLoadConfig(Config):
    epochs = 50
    # TODO: change to your file name
    weights_file = 'model_trained.h5'

    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/control/1'


class RealData_1_percent(Config):
    epochs = 50
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/control/1'


class RealData_5_percent(Config):
    epochs = 50
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.05/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.05/control/1'


class RealData_10_percent(Config):
    epochs = 50
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.1/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.1/control/1'


class MNIST_config(Config):
    ###################################################
    # Note that MNIST doesn't require images folder   #
    # since dataset is downloaded from Net. Instead,  #
    # you need to pass `zeros_in_bag` and `zero_bags` #
    # parameters.                                     #
    ###################################################
    healthy_dir, diseased_dir = None, None
    epochs = 50
    batch_size = 5
    bag_size = 100
    # Percentage of zeros in one bag
    zeros_in_bag = 0.05
    # Percentage of bags labeled as '0' in all dataset
    zero_bags = 0.5


class MNIST_1_percent(MNIST_config):
    zeros_in_bag = 0.01
    bag_size = 300
    batch_size = 1


class MNIST_5_percent(MNIST_config):
    zeros_in_bag = 0.05

class MNIST_10_percent(MNIST_config):
    zeros_in_bag = 0.10

class MNIST_20_percent(MNIST_config):
    zeros_in_bag = 0.20


class MNIST_50_percent(MNIST_config):
    zeros_in_bag = 0.50


CONFIG_TYPES = {'debug': DebugConfig,
                'test': TestConfig,
                'production': ProductionConfig,
                'production_load': ProductionLoadConfig,
                'real_1': RealData_1_percent,
                'real_5': RealData_5_percent,
                'real_10': RealData_10_percent,
                'mnist': MNIST_config,
                'mnist_1': MNIST_1_percent,
                'mnist_5': MNIST_5_percent,
                'mnist_10': MNIST_10_percent,
                'mnist_20': MNIST_20_percent,
                'mnist_50': MNIST_50_percent,
                }


def load_config(args):
        config = CONFIG_TYPES.get(args.config_type, None)
        if not config:
            raise ValueError('Incorrect type of config. ' +
                             f'Should be one of following: {", ".join(CONFIG_TYPES.keys())}')
        return config
