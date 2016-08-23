from unittest import TestCase
from confusion_matrix import ConfusionMatrix
import numpy as np


class TestConfusionMatrix(TestCase):

    def setUp(self):
        label_names = ['banana', 'apple', 'orange']
        labels = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1]
            ]
        )
        predictions = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.7, 0.1, 0.2],
                [0.4, 0.3, 0.3],
                [0.2, 0.6, 0.2],
                [0.5, 0.2, 0.3],
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 0.7],
                [0.1, 0.8, 0.1],
                [0.3, 0.5, 0.2],
                [0.1, 0.0, 0.9],
                [0.0, 1.0, 0.0]
            ]
        )
        self.confusion_matrix = ConfusionMatrix(predictions, labels, class_names=label_names)

    def tearDown(self):
        del self.confusion_matrix

    def test_confusion_matrix_not_normalized(self):
        expected = np.array(
            [
                [2, 1, 1],
                [1, 2, 0],
                [1, 1, 2]
            ]
        )
        actual = self.confusion_matrix.matrix_not_normalized
        are_equal = np.array_equal(actual, expected)
        self.assertEqual(are_equal, True)

    def test_confusion_matrix_normalized(self):
        expected = np.array(
            [
                [0.50, 0.25, 0.25],
                [0.33, 0.67, 0.00],
                [0.25, 0.25, 0.50]
            ]
        )
        actual = self.confusion_matrix.normalized
        actual = actual.round(decimals=2)
        are_equal = np.array_equal(actual, expected)
        self.assertEqual(are_equal, True)

