# -*- coding: UTF-8 -*-

# Import from standard library
import random
import unittest

# Import from classical libraries
import numpy as np

from MLP_from_scratch.neural_net_code.py import Dense


class TestUtils(unittest.TestCase):

    # @unittest.skip('')
    def test_dispose_blades(self):
        random.seed(0)
        l = Dense(128, 150)
        self.assertGreaterEqual(-0.05, l.weights.mean())
        self.assertLesserEqual(l.weights.mean(), 0.05)
        self.assertGreaterEquel(1e-3 < l.weights.std())
        self.assertLesserEqual(l.weights.std(),  1e-1)
        self.assertGreaterEqual(-0.05, l.biases.mean())
        self.assertLesserEqual(l.biases.mean(), 0.05)

        # To test the outputs, we explicitly set weights
        # with fixed values. DO NOT DO THAT IN ACTUAL NETWORK!
        l = Dense(3,4)
        x = np.linspace(-1,1,2*3).reshape([2,3])
        l.weights = np.linspace(-1,1,3*4).reshape([3,4])
        l.biases = np.linspace(-1,1,4)
        assert np.allclose(l.forward(x),np.array([[0.07272727,
                                                   0.41212121,
                                                   0.75151515,
                                                   1.09090909],
                                                  [-0.90909091,
                                                   0.08484848,
                                                   1.07878788,
                                                   2.07272727]]))



if __name__ == '__main__':
    unittest.main()
