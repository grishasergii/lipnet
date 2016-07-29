from unittest import TestCase
import numpy as np
from confusion_matrix import ConfusionMatrix, ConfusionTable


class TestConfusionTable(TestCase):

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

    def test_banana_true_positive(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['banana'].true_positive
        self.assertEqual(actual, expected)

    def test_banana_true_negative(self):
        expected = 5
        actual = self.confusion_matrix.confusion_tables['banana'].true_negative
        self.assertEqual(actual, expected)

    def test_banana_false_positive(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['banana'].false_positive
        self.assertEqual(actual, expected)

    def test_banana_false_negative(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['banana'].false_negative
        self.assertEqual(actual, expected)

    def test_apple_true_positive(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['apple'].true_positive
        self.assertEqual(actual, expected)

    def test_apple_true_negative(self):
        expected = 6
        actual = self.confusion_matrix.confusion_tables['apple'].true_negative
        self.assertEqual(actual, expected)

    def test_apple_false_positive(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['apple'].false_positive
        self.assertEqual(actual, expected)

    def test_apple_false_negative(self):
        expected = 1
        actual = self.confusion_matrix.confusion_tables['apple'].false_negative
        self.assertEqual(actual, expected)

    def test_orange_true_positive(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['orange'].true_positive
        self.assertEqual(actual, expected)

    def test_orange_true_negative(self):
        expected = 6
        actual = self.confusion_matrix.confusion_tables['orange'].true_negative
        self.assertEqual(actual, expected)

    def test_orange_false_positive(self):
        expected = 1
        actual = self.confusion_matrix.confusion_tables['orange'].false_positive
        self.assertEqual(actual, expected)

    def test_orange_false_negative(self):
        expected = 2
        actual = self.confusion_matrix.confusion_tables['orange'].false_negative
        self.assertEqual(actual, expected)
