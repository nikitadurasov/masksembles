import unittest
import tensorflow as tf

import masksembles.keras


class TestCreation(unittest.TestCase):

    def test_init_failed(self):
        layer = masksembles.keras.Masksembles2D(4, 11.)
        self.assertRaises(ValueError, layer, tf.ones([4, 10, 4, 4]))

    def test_init_success(self):
        layer = masksembles.keras.Masksembles2D(4, 11.)
