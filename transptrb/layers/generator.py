import theano
import numpy as np


def gen_param(name, size, rng=None):
    assert name in ['W', 'b', 'hbias', 'vbias']
    assert len(size) > 0

    if rng is None:
        rng = np.random.RandomState(888)

    if len(size) == 1:
        init_param = theano.shared(
            value=np.zeros(
                shape=size,
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    else:
        init_param = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / np.sum(size)),
                    high=np.sqrt(6. / np.sum(size)),
                    size=size
                ),
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

    return init_param
