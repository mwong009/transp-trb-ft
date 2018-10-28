#!/usr/bin/python3
# -*- coding: utf-8 -*-
import packageloader

import os
import sys
import timeit
import pickle
import gzip

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from models import MLP
from optimizers import SGD, Nadam
from train import load_data


def main(base_lr=0.001, base_lr_decay=0.002, dataset='mnist.pkl.gz',
         n_epochs=10, batch_size=512, L1_reg=0.00, L2_reg=0.0001,
         n_hidden=[500], pretraining_epochs=0, n_chains=5, dropout=0.5,
         dropconnect=0., rbm_dropout=0.5):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # data
    y = T.ivector('y')  # labels
    is_train = T.iscalar('is_train')
    learning_rate = T.scalar('learning_rate')

    mlp = MLP(
        input=x,
        output=y,
        n_in=28 * 28,
        hidden_layers_sizes=[500],
        n_out=10,
        dropout=dropout,
        dropconnect=dropconnect,
        optimizer=Nadam,
        learning_rate=learning_rate,
        is_train=is_train
    )

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=mlp.errors,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: T.cast(0, 'int32')
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=mlp.errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: T.cast(0, 'int32')
        }
    )

    train_model = theano.function(
        inputs=[index, learning_rate],
        outputs=mlp.cost,
        updates=mlp.updates(),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: T.cast(1, 'int32')
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')

    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        lr = np.maximum(
            1e-6 * base_lr, base_lr * ((1. - base_lr_decay) ** epoch),
            dtype=theano.config.floatX)
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index, lr)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%,' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.,
                    ),
                    'learning rate %f' % lr
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(mlp, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    print('... tests OK')

if __name__ == '__main__':
    main()
