import unittest
from utils import *
import numpy as np
import os


class BasicUtilsTest(unittest.TestCase):

    def setUp(self):
        if os.path.exists('test_only'):
            os.rmdir('test_only')

    def tearDown(self):
        if os.path.exists('test_only'):
            os.rmdir('test_only')

    def test_get_best_bag_size(self):
        self.assertEqual(3, get_best_bag_size(
            np.ones(4*28*28*1).reshape((4, 28, 28, 1)),
            np.ones(5*28*28*1).reshape((5, 28, 28, 1))
        ))
        self.assertEqual(5, get_best_bag_size(
            np.ones(4*28*28*1).reshape((4, 28, 28, 1)),
            np.ones(6*28*28*1).reshape((6, 28, 28, 1))
        ))
        self.assertEqual(4, get_best_bag_size(
            np.ones(3*28*28*1).reshape((3, 28, 28, 1)),
            np.ones(5*28*28*1).reshape((5, 28, 28, 1))
        ))

    def test_ensure_folder_create(self):
        ensure_folder('test_only')
        try:
            os.listdir('test_only')
        except:
            self.fail('Directory was not created!')

    def test_ensure_folder_do_not_create_if_exists(self):
        os.mkdir('test_only')
        ensure_folder('test_only')
        self.assertTrue(os.path.exists('test_only'))
