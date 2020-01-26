import numpy as np
from itertools import product
import pandas as pd
from collections import OrderedDict
from . import gaussian_process as gp
from . import space

gaussian_process = gp


def random(metaparamspace, allow_interpolation_override=None):
    """selects metaparams randomly"""
    params = {}
    for key in metaparamspace:
        value_list = list(space.unroll_1d(metaparamspace[key]))
        if len(value_list) == 0:
            raise ValueError(f'Meta space dimension "{key}" cannot be empty.')
        params[key] = value_list[np.random.choice(len(value_list))]
    return params



def _rm_duplicates(values):
    return list(OrderedDict.fromkeys(values))


# https://arxiv.org/abs/1605.07079
# http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
