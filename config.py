class Config:
    """
    Basic config
    """
    work_dir = 'trained'
    save_best_only = True
    epochs = 10
    verbose = False
    debug = False
    load_from = None
    tensorboard_dir = 'tlogs'
    bag_size = 'auto'
    diseased_dir = 'diseased_imgs'
    healthy_dir = 'healthy_imgs'


class DebugConfig(Config):
    debug = True
    verbose = True
    load_from = 'model_trained.h5'
    diseased_dir = '/Users/akoyharin/Desktop/diseased_debug'
    healthy_dir = '/Users/akoyharin/Desktop/healthy_debug'


class ProductionConfig(Config):
    debug = True
    epochs = 10
    diseased_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/diseased/1'
    healthy_dir = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_claassen_1/corino/Scratch/EmanuelDatasets/tryp_0.01/control/1'
