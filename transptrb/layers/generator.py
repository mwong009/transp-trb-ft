import theano
import numpy as np


def gen_param(name, shape, rng=None):
    assert name in ['W', 'b', 'hbias', 'vbias']
    assert len(shape) > 0

    if rng is None:
        rng = np.random.RandomState(888)

    if len(shape) == 1:
        init_param = theano.shared(
            value=np.zeros(
                shape=shape,
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    else:
        init_param = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / np.sum(shape)),
                    high=np.sqrt(6. / np.sum(shape)),
                    size=shape
                ),
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    return init_param
