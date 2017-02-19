import mnist_softmax
import unittest

class MnistSoftmaxTest(unittest.TestCase):
    """MNIST softmax tests"""

    def test_find_accuracy(self):
        mnistSoftmax = mnist_softmax.MnistSoftmax()
        accuracy = mnistSoftmax.find_accuracy()

        # Test for acceptance. This should be roughly 92%
        self.assertTrue(bool(.90 < accuracy and accuracy < 0.95))
