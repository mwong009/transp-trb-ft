import os
import sys
import packageloader

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import gzip
import requests
import numpy as np

from train import load_data
from models import MLP
from layers import DenseLayer
from optimizers import SGD, Nadam, Momentum
from unittest import TestCase

url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"


class DataLoaderTest(TestCase):

    def setUp(self):
        self.dataset = os.path.join(
            os.path.split(__file__)[0],
            "mnist.pkl.gz"
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
