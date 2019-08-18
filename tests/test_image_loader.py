import unittest
import numpy as np
from image_loader import ImageLoader
import glob


class ImageLoaderTest(unittest.TestCase):

    def setUp(self):
        self.diseased_path, self.healthy_path = 'debug_imgs/diseased', 'debug_imgs/healthy'
        self.image_loader = ImageLoader(diseased_dir=self.diseased_path,
                                        healthy_dir=self.healthy_path,
                                        bag_size=10, batch_size=32)
        self.all_diseased, self.all_healthy = glob.glob(self.diseased_path), glob.glob(self.healthy_path)

    def test_add_color_channel(self):
        a = np.zeros(28*28*30).reshape((30, 28, 28))
        self.assertEqual((30, 28, 28, 1), ImageLoader.add_color_channel(a).shape)

    def test_load_paths(self):
        with self.subTest('Diseased paths'):
            paths = glob.glob('debug_imgs/diseased')
            self.assertEqual(sorted(paths), sorted(self.image_loader.diseased_paths))
        with self.subTest('Healthy paths'):
            paths = glob.glob('debug_imgs/healthy')
            self.assertEqual(sorted(paths), sorted(self.image_loader.healthy_paths))
