import os
import sys
import packageloader

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import gzip
import requests
import pytest
import numpy as np

from transptrb.train import load_data
from transptrb.models import MLP, DBN
from transptrb.layers import DenseLayer
from transptrb.optimizers import SGD, Nadam, Momentum
from unittest import TestCase

url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"


class DataLoaderTest(TestCase):

    def setUp(self):
        self.dataset = os.path.join(
            os.path.split(__file__)[0],
            "../.cache/mnist.pkl.gz"
        )
        if not os.path.isfile(self.dataset):
            r = requests.get(url)

            with open(self.dataset, 'wb') as f:
                f.write(r.content)

    def test_loaddataset(self):
        datasets = load_data(self.dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        self.assertEqual(datasets[0][0], train_set_x, 'Incorrect index slice')
        self.assertEqual(datasets[2][1], test_set_y, 'Incorrect index slice')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class ModelTest(TestCase):
    def setUp(self):
        dataset = os.path.join(
            os.path.split(__file__)[0],
            "../.cache/mnist.pkl.gz"
        )
        if not os.path.isfile(dataset):
            r = requests.get(url)

            with open(dataset, 'wb') as f:
                f.write(r.content)

        self.batch_size = 20
        self.n_in = 100
        self.hidden_layers_sizes = [50, 50]
        self.n_out = 10

        self.rng = np.random.RandomState(888)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.input = theano.shared(
            value=np.asarray(
                self.rng.uniform(0., 1., size=(self.batch_size, self.n_in)),
                dtype=theano.config.floatX
            ),
            name='input',
            borrow=True
        )

        self.output = T.cast(theano.shared(
            value=np.asarray(
                np.random.randint(0, 9, size=(self.batch_size,)),
                dtype=theano.config.floatX
            ),
            name='output',
            borrow=True
        ), 'int32')

        self.datasets = load_data(dataset)

    def test_createDBN(self):

        x = T.matrix('x')   # data
        y = T.ivector('y')  # labels
        is_train = T.iscalar('is_train')

        dbn = DBN(
            input=self.input,
            output=self.output,
            n_in=self.n_in,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_out=self.n_out,
            is_train=is_train
        )

        t, v = dbn.build_pretraining_functions(self.datasets, self.batch_size)
        t, v, s = dbn.build_finetune_functions(self.datasets, self.batch_size)

    def test_createMLP(self):

        x = T.matrix('x')   # data
        y = T.ivector('y')  # labels
        is_train = T.iscalar('is_train')

        mlp = MLP(
            input=self.input,
            output=self.output,
            n_in=self.n_in,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_out=self.n_out,
            is_train=is_train
        )

        mlp_output = mlp.output
