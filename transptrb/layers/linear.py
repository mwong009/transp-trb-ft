#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np

from ..layers.generator import gen_param


class LinearRegression(object):
    def __init__(self, input, n_in, n_out):

        self.W = gen_param(name='W', shape=(n_in, n_out))
        self.b = gen_param(name='b', shape=(n_out,))

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
            return T.mean((self.output - y) ** 2, axis=0)
        else:
            raise NotImplementedError()

    def r_mean_squared_error(self, y):
        mse = self.mean_squared_error(y)
        return T.sqrt(mse)

    def mean_absolute_error(self, y):
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.output',
                ('y', y.type, 'output', self.output.type)
            )

        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            return T.mean(T.abs_(self.output - y), axis=0)
        else:
            raise NotImplementedError()

