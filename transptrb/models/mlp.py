#!/usr/bin/python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from ..layers import DenseLayer, LogisticRegression
from ..optimizers import SGD


class MLP(object):
    def __init__(self, input, output, n_in, hidden_layers_sizes, n_out,
                 dropout=None, dropconnect=None, optimizer=SGD,
                 learning_rate=0.1, is_train=0):

        self.dense_layers = []
        self.params = []
        self.consider_constants = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        self.rng = np.random.RandomState(888)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_in
                layer_input = input
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.dense_layers[-1].output

            dense_layer = DenseLayer(
                input=layer_input,
                rng=self.rng,
                theano_rng=self.theano_rng,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.softplus,
                dropout=dropout,
                is_train=is_train
            )

            self.dense_layers.append(dense_layer)
            self.params.extend(dense_layer.params)

        self.log_regression_layer = LogisticRegression(
            input=self.dense_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )

        self.params.extend(self.log_regression_layer.params)

        self.cost = self.log_regression_layer.negative_loglikelihood(output)
        self.errors = self.log_regression_layer.errors(output)

        self.input = input
        self.output = output
        self.learning_rate = learning_rate

        # model updates
        self.opt = optimizer(self.params)

    def updates(self):
        return self.opt.update(
            self.cost,
            self.params,
            self.learning_rate,
            self.consider_constants
        )
