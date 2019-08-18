from model import build_model
from keras.preprocessing.image import ImageDataGenerator
from model_config import Config as config

if __name__ == '__main__':
    generator = ImageDataGenerator(rescale=1/255.).flow_from_directory(config.images_directory)
    model = build_model(image_shape=generator.image_shape, config=config)
    model.fit_generator(generator=generator, )