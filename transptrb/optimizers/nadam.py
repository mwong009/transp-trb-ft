#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np


class Nadam(object):
    def __init__(self, params, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 schedule_decay=0.004):
        assert (beta_1 >= 0. and beta_1 < 1.)
        assert (beta_2 >= 0. and beta_2 < 1.)
        assert (schedule_decay >= 0. and schedule_decay < 1.)

        self._iterations = theano.shared(
            value=np.zeros_like(0., dtype=theano.config.floatX),
            name='iterations'
        )
        self._schedule = theano.shared(
            value=np.ones_like(1., dtype=theano.config.floatX),
            name='schedule'
        )

        self._ms = []
        self._vs = []
        for param in params:
            self._ms.append(theano.shared(
                value=np.zeros_like(
                    param.get_value(),
                    dtype=theano.config.floatX
                ),
                name='ms'
            ))
            self._vs.append(theano.shared(
                value=np.zeros_like(
                    param.get_value(),
                    dtype=theano.config.floatX
                ),
                name='vs'
            ))

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

    def update(self, cost, params, learning_rate, consider_constants=None):
        grads = [T.grad(cost, param, consider_constants) for param in params]

        iterations = self._iterations
        schedule = self._schedule
        ms = self._ms
        vs = self._vs

        updates = []
        updates.append((iterations, iterations + 1))

        t = iterations + 1.
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            T.pow(T.cast(0.96, 'floatX'), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            T.pow(T.cast(0.96, 'floatX'), (t + 1.) * self.schedule_decay)))
        m_new = schedule * momentum_cache_t
        m_next = schedule * momentum_cache_t * momentum_cache_t_1

        updates.append((schedule, m_new))

        for param, grad, m, v in zip(params, grads, ms, vs):
            grad_prime = grad / (1. - m_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * grad
            m_t_prime = m_t / (1. - m_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * grad * grad
            v_t_prime = v_t / (1. - T.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * grad_prime + (
                momentum_cache_t_1 * m_t_prime)

            param_t = (
                param - learning_rate * m_t_bar /
                (T.sqrt(v_t_prime) + self.epsilon)
            )

            updates.append((m, m_t,))
            updates.append((v, v_t))
            updates.append((param, param_t))

        return updates
