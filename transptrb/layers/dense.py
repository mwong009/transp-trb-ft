#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import numpy as np

from ..layers.generator import gen_param


class DenseLayer(object):
    def __init__(self, input, rng, theano_rng, n_in, n_out, W=None, b=None,
                 activation=None, dropout=None, dropconnect=None,
                 is_train=0, resnet_input=None):

        if W is None:
            W = gen_param(name='W', shape=(n_in, n_out), rng=rng)

        if b is None:
            b = gen_param(name='b', shape=(n_out,))

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        if resnet_input is not None:
            lin_output = lin_output + resnet_input
        output = activation(lin_output)

        if (dropout is not None) and (dropout > 0.):
            output = ifelse(
                condition=T.eq(is_train, 1),
                then_branch=T.switch(
                    theano_rng.binomial(
                        size=(n_out,),
                        p=(1.-dropout),
                        dtype=theano.config.floatX
                    ),
                    output, 0.
                ),
                else_branch=output*(1.-dropout)
            )

        self.output = output

        self.params = [self.W, self.b]
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.is_train = is_train
