#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np


class SGD(object):
    def __init__(self, params):
        pass

    def update(self, cost, params, learning_rate, consider_constants):
        grads = [T.grad(cost, param, consider_constants) for param in params]

        updates = []
        for param, grad in zip(params, grads):
            update = param - learning_rate * grad
            updates.append((param, update))

        return updates
