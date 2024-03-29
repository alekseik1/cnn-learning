# Image recognition with "bag" model
This repo contains CNN model based on _Autoencoder_ and _Classifier_.
The model uses "bags" as additional dimension to support weakly-supervised structure of data.

## Contents
1. [Preparation](#preparing)
2. [How to run](#running)
3. [Select images](#dataset-path)
    - [MNIST](#mnist-dataset)
    - [From folder](#load-from-folder)
4. [Custom configs](#extending-configs)
5. [Model overview](#architecture)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
    - [Classifier](#classifier)
6. [Known problems](#known-problems)
7. [Contributing](#contributing)

## Preparing
__On ETH cluster__, all required packages are already included. 
You __don't__ need to create any virtual environments as it causes problems.

If you run it outside ETH cluster, I recommend you to create virtual environment for that.

#### virtualenv
```bash
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Anaconda (or conda)
```bash
conda create --name cnn_bags
conda activate cnn_bags
conda install --file requirements.txt
```
You are ready to rock!


## Running
To run model, you need to provide both __config__ and __label__ as command line options.

The script will create folder with name `your_label` and subfolders `model` and `tesnorflow_logs`.

All configuration (except labels) is done via configuration file `config.py`.
You need to load appropriate config, e.g.:
```bash
python pipeline.py --config mnist_1 --label test_run_only
```
Here `mnist_1` is a config name.
You can see all available config names by running:
```bash
python pipeline.py --help
```

## Dataset path
The model can use either images from folder or built-in MNIST dataset.

### MNIST dataset
Currently, there are 4 types of configs for MNIST:

| Name       | Meaning                            |
|------------|------------------------------------|
| `mnist_1`  | 1% of zeros in one 'diseased' bag  |
| `mnist_5`  | 5% of zeros in one 'diseased' bag  |
| `mnist_10` | 10% of zeros in one 'diseased' bag |
| `mnist_20` | 20% of zeros in one 'diseased' bag |

Choose appropriate config, then:

```bash
python pipeline -c mnist_10 -l my_first_mnist_experiment
```


### Load from folder
Before loading, you need to correct paths in desired config.
Take a look at `config.py`, find config that suits you best and __don't forget to change paths if necessary__.
_By default, all configs except `debug` take images from Corin's folder on NAS._

Once you are edited config, run model:
```bash
python pipeline.py --config <your_config> --label <your_label>
```


### Extending configs
It's easy to write your own configuration. 
Just create a class in `config.py` inheriting `Config` class and make required changes.

## Architecture
You can always look up `model.py`, function `_create_model` to see it.
For now, architecture is as follows:

#### Encoder
- Conv2D(64, (3, 3), activation='relu', padding='same')
- Conv2D(128, (3, 3), activation='relu', padding='same')
- MaxPooling2D((2, 2), padding='same', strides=2)
- Conv2D(128, (3, 3), activation='relu', padding='same')
- MaxPooling2D((2, 2), padding='same', strides=2)
- Conv2D(64, (3, 3), activation='relu', padding='same')
- Conv2D(8, (3, 3), activation='relu', padding='same')


#### Decoder
- Conv2D(128, (3, 3), activation='relu', padding='same')
- UpSampling2D((2, 2))
- Conv2D(64, (3, 3), activation='relu', padding='same')
- UpSampling2D((2, 2))
- Conv2D(32, (3, 3), activation='relu', padding='same')
- Conv2D(input_shape[-1], (3, 3), activation='relu', padding='same')

#### Classifier
- Dense(128, activation=\<from config\>)
- Dropout(rate=0.5)
- Dense(1, activation=\<from config\>)

## Known problems
- Easy OOMs even on Tesla GPUs (with 32Gb of RAM).
Use `batch_size=1` to avoid such problems (_will take long to train_).
- Poorly decoded images. _To be investigated..._
- (as consequence) Bad results for real cells.
Maybe we should additionally do some preprocessing of images?
- Slow image loading from NAS.
Perhaps, due to it's network-based nature.

## Contributing
PRs are always welcome.
If you'd like to become the maintainer, email [me](mailto:1alekseik1@gmail.com).
