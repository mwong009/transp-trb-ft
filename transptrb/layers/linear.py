#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np

from layers.generator import gen_param


class LinearRegression(object):
    def __init__(self, input, n_in):

        self.W = gen_param(name='W', shape=(n_in,))
        self.b = gen_param(name='b', shape=(1,))

        self.output = T.dot(input, self.W) + self.b

        self.params = [self.W, self.b]
        self.input = input

    def mean_squared_error(self, y):
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.output',
                ('y', y.type, 'output', self.output.type)
            )

        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean((self.output - y) ** 2)
        else:
            raise NotImplementedError()

    def rmse(self, mse):
        return T.sqrt(mse)
