import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class KerasImageLoader:

    # TODO: feature to augment with rotations/flips
    def __init__(self, classes_directory,
                 bag_size: int=10,
                 batch_size: int=10):
        self.classes_directory = classes_directory
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.image_generator = ImageDataGenerator(rescale=1/255.)
        self.image_flow = self.image_generator\
            .flow_from_directory(classes_directory,
                                 class_mode='binary',
                                 batch_size=int(2*bag_size*batch_size))
        self.image_cache = {0: [], 1: []}
        self.total_images = self.image_flow.n
        self.image_shape = self.image_flow.image_shape

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def cache_surplus(self, array, label):
        last_index = -(len(array) % self.bag_size)
        if last_index == 0:
            # If no need to cache anything
            return array
        to_cache = array[last_index:]
        self.image_cache.get(label).extend(to_cache)
        return array[:last_index]

    def handle_division_problems(self, array, label):
        result = self.augemnt_or_cache(array, label)
        # TODO: can potentially loop forever!
        while result.shape[0] == 0:
            result = self.augemnt_or_cache(array, label)
        return result

    def augemnt_or_cache(self, array, label):
        to_trim = len(array) % self.bag_size
        to_augment = self.bag_size - to_trim
        cache = self.image_cache.get(label)
        if len(cache) >= to_augment:
            # If where are enough images in cache
            cache_slice = np.array([cache.pop() for _ in range(to_augment)])
            array = np.concatenate((array, cache_slice))
        else:
            # If images are not enough, we just trim batch and save extra pics to cache
            array = self.cache_surplus(array, label)
        return array

    def get_cache_info(self):
        return len(self.image_cache[0]), len(self.image_cache[1])

    def next(self):
        # TODO: track how many images are loaded and stop at the right moment
        # We get a batch
        keras_batch, keras_labels = next(self.image_flow)
        keras_batch = keras_batch if len(keras_batch.shape) == 4 else keras_batch.reshape((*keras_batch.shape, 1))
        positive_pics = keras_batch[np.argwhere(keras_labels == 1).reshape(-1)]
        positive_pics = self.handle_division_problems(positive_pics, 1)
        positive_pics = positive_pics.reshape((-1, self.bag_size, *positive_pics.shape[1:]))
        negative_pics = keras_batch[np.argwhere(keras_labels == 0).reshape(-1)]
        negative_pics = self.handle_division_problems(negative_pics, 0)
        negative_pics = negative_pics.reshape((-1, self.bag_size, *negative_pics.shape[1:]))
        result_x = np.concatenate((positive_pics, negative_pics))
        p = np.random.permutation(len(result_x))
        result_y = np.concatenate((np.ones(len(positive_pics)), np.zeros(len(negative_pics))))
        return result_x[p], {'classifier_output': result_y[p], 'decoded_output': result_x[p]}


if __name__ == '__main__':
    loader = KerasImageLoader(classes_directory='debug_imgs', batch_size=2, bag_size=10)
    for i in range(1000):
        print('shape: {}'.format(loader.next()[0].shape))
