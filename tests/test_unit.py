import packageloader

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from transptrb.models import MLP
from transptrb.layers import DenseLayer
from transptrb.optimizers import SGD, Nadam, Momentum
from unittest import TestCase


class OptimizerTest(TestCase):

    def setUp(self):
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


class LayerTest(TestCase):
    def setUp(self):
        batch_size = 40
        self.n_in = 10
        self.n_out = 100

        self.rng = np.random.RandomState(888)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.input = theano.shared(
            value=np.random.normal(0., 1., size=(batch_size, self.n_in)),
            name='input'
        )

    def test_denseLayer(self):

        dense_layer = DenseLayer(
            rng=self.rng,
            theano_rng=self.theano_rng,
            input=self.input,
            n_in=self.n_in,
            n_out=self.n_out,
            activation=T.nnet.softplus,
            dropout=None,
            is_train=1
        )
