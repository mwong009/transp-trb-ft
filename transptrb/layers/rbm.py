#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from layers.generator import gen_param
from optimizers import SGD


class RBM(object):
    def __init__(self, input, rng=None, theano_rng=None, n_visible=None,
                 n_hidden=None, W=None, hbias=None, vbias=None, dropout=None,
                 v_activation=T.nnet.sigmoid, h_activation=T.nnet.sigmoid,
                 optimizer=SGD, is_train=0):

        if rng is None:
            rng = np.random.RandomState(888)

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        if W is None:
            W = gen_param(name='W', shape=(n_visible, n_hidden), rng=rng)

        if hbias is None:
            hbias = gen_param(name='hbias', shape=(n_hidden,))

        if vbias is None:
            vbias = gen_param(name='vbias', shape=(n_visible,))

        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.dropout = dropout
        self.is_train = is_train

        self.input = input
        self.theano_rng = theano_rng
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.params = [self.W, self.hbias, self.vbias]

        # model updates
        self.opt = optimizer(self.params)

        self.v_activation = v_activation
        self.h_activation = h_activation

    def sample(self, pre_activation, mean, dist=None):

        assert dist in ['binomial', 'normal']

        if dist == 'binomial':
            sample = self.theano_rng.binomial(
                size=pre_activation.shape,
                n=1,
                p=mean,
                dtype=theano.config.floatX
            )
        elif dist == 'normal':
            sample = self.theano_rng.normal(
                size=mean.shape,
                avg=pre_activation,
                std=T.nnet.sigmoid(pre_activation),
                dtype=theano.config.floatX
            )

        return sample

    def propup(self, v):

        pre_activation = T.dot(v, self.W) + self.hbias
        activation = self.h_activation(pre_activation)
        return [pre_activation, activation]

    def sample_h_given_v(self, v0_sample):

        pre_activation, h1_mean = self.propup(v0_sample)

        if self.h_activation is T.nnet.sigmoid:
            h1_sample = self.sample(pre_activation, h1_mean, 'binomial')

        elif self.h_activation is T.nnet.softplus:
            normal_sample = self.sample(pre_activation, h1_mean, 'normal')
            h1_sample = T.nnet.softplus(normal_sample)

        if (self.dropout is not None) and (self.dropout > 0.):
            h1_sample = theano.ifelse.ifelse(
                condition=T.eq(self.is_train, 1),
                then_branch=T.switch(
                    self.sample(h1_sample, (1.-self.dropout), 'binomial'),
                    h1_sample, 0.
                ),
                else_branch=h1_sample*(1.-self.dropout)
            )

        return[pre_activation, h1_mean, h1_sample]

    def propdown(self, h):

        linalg = T.dot(h, self.W.T) + self.vbias
        a = self.v_activation(linalg)
        return [linalg, a]

    def sample_v_given_h(self, h0_sample):

        pre_activation, v1_mean = self.propdown(h0_sample)

        if self.v_activation is T.nnet.sigmoid:
            v1_sample = self.sample(pre_activation, v1_mean, 'binomial')

        elif self.v_activation is T.nnet.softplus:
            normal_sample = self.sample(pre_activation, v1_mean, 'normal')
            h1_sample = T.nnet.softplus(normal_sample)

        return[pre_activation, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):

        v1_pre, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_pre, h1_mean, h1_sample = self.sample_h_given_v(v1_mean)
        return [v1_pre, v1_mean, v1_sample,
                h1_pre, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):

        h1_pre, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_pre, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [h1_pre, h1_mean, h1_sample,
                v1_pre, v1_mean, v1_sample]

    def free_energy(self, v):

        wx_b = T.dot(v, self.W) + self.hbias
        visible_term = T.dot(v, self.vbias)
        # visible_term = -0.5*T.sum(T.sqr(v - self.vbias), axis=1)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - visible_term

    def get_valid_error(self):

        ph_pre, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        v1_pre, v1_mean, v1_sample = self.sample_v_given_h(ph_mean)

        return T.mean(T.sqr(self.input - v1_mean))

    def get_cost_updates(self, learning_rate, persistent=None, k=1):

        ph_pre, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([nv_pres, nv_means, nv_samples, nh_pres, nh_means, nh_samples],
            gibbs_updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        chain_end = nv_means[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        consider_constants = [chain_end]

        updates = self.opt.update(
            cost, self.params, learning_rate, consider_constants
        )

        for param, update in updates:
            gibbs_updates[param] = update

        if persistent:
            gibbs_updates[persistent] = nh_samples[-1]

        monitoring_cost = T.mean(T.sqr(self.input - chain_end))

        return monitoring_cost, gibbs_updates
