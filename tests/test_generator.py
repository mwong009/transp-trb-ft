import packageloader

from unittest import TestCase
from transptrb.layers.generator import gen_param

import numpy as np


class GeneratorTest(TestCase):

    def test_genParam(self):
        rng = np.random.RandomState(888)
        shape = (20, 2)
        param = gen_param('W', shape, rng)

        assert param.get_value().shape == shape
