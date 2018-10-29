#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np


class DenseLayer(object):
    def __init__(self, input, rng, theano_rng, n_in, n_out, W=None, b=None,
                 activation=None, dropout=None, dropconnect=None,
                 is_train=0):

        if W is None:
            W = gen_param(name='W', shape=(n_in, n_out), rng=rng)

        if b is None:
            b = gen_param(name='b', shape=(n_out,))

        self.W = W
        self.b = b

        if (dropconnect is None) or (dropconnect == 0.):
            lin_output = T.dot(input, self.W) + self.b
            output = activation(lin_output)
            self.consider_constant = None

        else:
            output = theano.ifelse.ifelse(
                condition=T.eq(is_train, 1),
                then_branch=activation(T.dot(input, T.switch(
                    theano_rng.binomial(
                        size=(n_in, n_out),
                        p=(1.-dropconnect),
                        dtype=theano.config.floatX
                    ),
                    self.W, 0.
                )) + self.b),
                else_branch=activation(
                    T.dot(input, (1.-dropconnect)*self.W) + self.b)
                # else_branch=activation(
                #   T.mean(normal_sample, axis=0) + self.b)
            )
            self.consider_constant = None

        if (dropout is not None) and (dropout > 0.):
            output = theano.ifelse.ifelse(
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
