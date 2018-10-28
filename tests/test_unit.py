import packageloader

import theano
import theano.tensor as T

import numpy as np

from models import MLP
from optimizers import SGD, Nadam, Momentum
from unittest import TestCase


class OptimizerTest(TestCase):
    def __init__(self, *args, **kwargs):
        super(OptimizerTest, self).__init__(*args, **kwargs)
        self.rng = np.random.RandomState(888)
        n_in = 10
        n_out = 100

        W = theano.shared(
            value=np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        b = theano.shared(
            value=np.zeros(
                shape=(n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.params = [W, b]

    def test_sgd(self):
        sgd = SGD(self.params)

    def test_nadam(self):
        nadam = Nadam(self.params)

    def test_momentum(self):
        momentum = Momentum(self.params)
