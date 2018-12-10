import theano
import numpy as np


def gen_param(name, shape, rng=None):
    assert name in ['W', 'b', 'hbias', 'vbias']
    assert len(shape) > 0

    if rng is None:
        rng = np.random.RandomState(888)

    if len(shape) == 1:
        parameter = theano.shared(
            value=np.zeros(
                shape=shape,
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    else:
        parameter = theano.shared(
            value=np.asarray(
                rng.normal(0, (1./shape[0]), size=shape),
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    return parameter
