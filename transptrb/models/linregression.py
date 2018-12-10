#!/usr/bin/python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from ..layers import DenseLayer, LinearRegression
from ..optimizers import SGD


class MultiLayerRegression(object):
    def __init__(self, input, output, n_in, n_hidden_sizes, n_out,
                 optimizer=SGD, is_train=0, learning_rate=0.1, dropout=None,
                 resnet=False):

        assert isinstance(n_hidden_sizes, list)

        self.layers = []
        self.params = []
        self.skip_connections = []
        self.n_layers = len(n_hidden_sizes)

        assert self.n_layers > 0

        self.rng = np.random.RandomState(888)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        for n in range(self.n_layers):
            if n == 0:
                input_size = n_in
                layer_input = input
                self.skip_connections.append(None)
            else:
                input_size = n_hidden_sizes[n - 1]
                layer_input = self.layers[-1].output

            dense_layer = DenseLayer(
                input=layer_input,
                rng=self.rng,
                theano_rng=self.theano_rng,
                n_in=input_size,
                n_out=n_hidden_sizes[n],
                activation=T.nnet.selu,
                dropout=dropout,
                is_train=is_train,
                resnet_input=self.skip_connections[-1]
            )

            if resnet:
                if n < (self.n_layers - 1):
                    if (n_hidden_sizes[n] == n_hidden_sizes[n+1]):
                        self.skip_connections.append(dense_layer.output)
                    else:
                        self.skip_connections.append(None)
                else:
                    self.skip_connections.append(None)

            self.layers.append(dense_layer)
            self.params.extend(dense_layer.params)

        self.regression_layer = LinearRegression(
            input=self.layers[-1].output,
            n_in=n_hidden_sizes[-1],
            n_out=n_out
        )

        self.params.extend(self.regression_layer.params)
        self.cost = T.mean(self.regression_layer.mean_squared_error(output))
        self.errors = self.regression_layer.r_mean_squared_error(output)

        self.input = input
        self.output = output
        self.learning_rate = learning_rate

        # model updates
        self.opt = optimizer(self.params)

        self.updates = self.opt.update(
            self.cost, self.params, self.learning_rate,
            consider_constants=None
        )
