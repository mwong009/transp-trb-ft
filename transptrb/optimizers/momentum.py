#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np


class Momentum(object):
    def __init__(self, params, momentum=0.9, epsilon=1e-7, nesterov=False):
        assert (momentum >= 0. and momentum < 1.)
        assert (nesterov is True or nesterov is False)

        self._velocity = [theano.shared(
            value=np.zeros_like(
                p.get_value(),
                dtype=theano.config.floatX),
            name=p.name + '_velocity') for p in params
        ]

        self.momentum = momentum
        self.nesterov = nesterov
        self.epsilon = epsilon

    def update(self, cost, params, learning_rate, consider_constants=None):
        grads = [T.grad(cost, param, consider_constants) for param in params]

        velocity = self._velocity

        updates = []
        for v, param, grad in zip(velocity, params, grads):
            updates.append((v, self.momentum * v - learning_rate * grad))
            update = updates[-1]

            if nesterov:
                inc = self.momentum * update - learning_rate * grad

            updates.append((param, inc))

        return updates
