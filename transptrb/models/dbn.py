#!/usr/bin/python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from layers import RBM, DenseLayer, LogisticRegression
from layers.generator import gen_param
from optimizers import SGD


class DBN(object):
    def __init__(self, input, output, n_in, hidden_layers_sizes, n_out,
                 dropout=None, optimizer=SGD,
                 is_train=0):

        self.dense_layers = []
        self.rbm_layers = []
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
                rng=self.rng,
                theano_rng=self.theano_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.softplus,
                dropout=dropout,
                is_train=is_train
            )

            rbm_layer = RBM(
                input=layer_input,
                rng=self.rng,
                theano_rng=self.theano_rng,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=dense_layer.W,
                hbias=dense_layer.b,
                dropout=dropout,
                h_activation=T.nnet.softplus,
                optimizer=optimizer,
                is_train=is_train
            )

            self.dense_layers.append(dense_layer)
            self.rbm_layers.append(rbm_layer)
            self.params.extend(dense_layer.params)

            if dense_layer.consider_constant is not None:
                self.consider_constants.extend(dense_layer.consider_constant)
            # end-for

        self.logistic_layer = LogisticRegression(
            input=self.dense_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )

        self.params.extend(self.logistic_layer.params)

        self.finetune_cost = self.logistic_layer.negative_loglikelihood(output)
        self.finetune_errors = self.logistic_layer.errors(output)

        self.input = input
        self.output = output
        self.is_train = is_train

        # model updates
        self.finetune_opt = optimizer(self.params)

    def _finetune_updates(self, learning_rate):
        return self.finetune_opt.update(
            self.finetune_cost,
            self.params,
            learning_rate,
            self.consider_constants
        )

    def build_pretraining_functions(self, datasets, batch_size, k=1):

        train_set_x = datasets[0][0]
        valid_set_x = datasets[1][0]

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('learning_rate')

        self.rbm_pretraining_fns = []
        self.rbm_pretraining_errors = []

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        for n, rbm_layer in enumerate(self.rbm_layers):

            persistent_chain = theano.shared(
                value=np.zeros(
                    shape=(batch_size, rbm_layer.n_hidden),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            rbm_cost, rbm_updates = rbm_layer.get_cost_updates(
                learning_rate, persistent_chain, k
            )

            train_rbm = theano.function(
                inputs=[index, learning_rate],
                outputs=rbm_cost,
                updates=rbm_updates,
                givens={
                    self.input: train_set_x[batch_begin: batch_end],
                    rbm_layer.is_train: T.cast(1, 'int32')
                },
                name='train_rbm' + '_' + str(n)
            )
            self.rbm_pretraining_fns.append(train_rbm)

            validate_rbm = theano.function(
                inputs=[index],
                outputs=rbm_layer.get_valid_error(),
                givens={
                    self.input: valid_set_x[batch_begin: batch_end],
                    rbm_layer.is_train: T.cast(0, 'int32')
                },
                name='valid_rbm' + '_' + str(n)
            )
            self.rbm_pretraining_errors.append(validate_rbm)
            # end-for

        return self.rbm_pretraining_fns, self.rbm_pretraining_errors

    def build_finetune_functions(self, datasets, batch_size):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('learning_rate')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        test_model = theano.function(
            inputs=[index],
            outputs=self.finetune_errors,
            givens={
                self.input: test_set_x[batch_begin: batch_end],
                self.output: test_set_y[batch_begin: batch_end],
                self.is_train: T.cast(0, 'int32')
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.finetune_errors,
            givens={
                self.input: valid_set_x[batch_begin: batch_end],
                self.output: valid_set_y[batch_begin: batch_end],
                self.is_train: T.cast(0, 'int32')
            }
        )

        train_model = theano.function(
            inputs=[index, learning_rate],
            outputs=self.finetune_cost,
            updates=self._finetune_updates(learning_rate),
            givens={
                self.input: train_set_x[batch_begin: batch_end],
                self.output: train_set_y[batch_begin: batch_end],
                self.is_train: T.cast(1, 'int32')
            }
        )

        return train_model, validate_model, test_model
