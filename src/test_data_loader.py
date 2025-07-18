import unittest
import tensorflow as tf
import numpy as np
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_data_loading(self):
        train_gen, val_gen = load_data(batch_size=16)
        self.assertIsInstance(train_gen, tf.keras.preprocessing.image.DirectoryIterator)
        self.assertIsInstance(val_gen, tf.keras.preprocessing.image.DirectoryIterator)
        
        batch = next(train_gen)
        images, labels = batch
        self.assertEqual(images.shape, (16, 224, 224, 3))
        self.assertEqual(labels.shape, (16, 10))
        
        print("Data loader test passed!")

if __name__ == '__main__':
    unittest.main()