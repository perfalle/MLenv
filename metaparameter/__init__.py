import numpy as np
from itertools import product
import pandas as pd
from collections import OrderedDict
#from . import gaussian_process as gp

#gaussian_process = gp


def random(metaparamspace):
    """selects metaparams randomly"""
    params = {}
    for key in metaparamspace:
        value_list = list(_unroll_1d(metaparamspace[key]))
        if len(value_list) == 0:
            raise ValueError(f'Meta space dimension "{key}" cannot be empty.')
        params[key] = value_list[np.random.choice(len(value_list))]
    return params


def space1d_list(values, allow_interpolation=False):
    return {
        'values': values,
        'allow_interpolation': allow_interpolation,
    }

def space1d_linear(minimum, maximum, steps=None, allow_interpolation=True):
    if steps is None and int(minimum) == minimum and int(maximum) == maximum:
        minimum = int(minimum)
        maximum = int(maximum)
        steps = maximum - minimum + 1
    return {
        'map_from_uniform': lambda u: u * (maximum - minimum) + minimum,
        'map_to_uniform': lambda x: (x - minimum) / (maximum - minimum),
        'steps': steps,
        'allow_interpolation': allow_interpolation,
    }

def space1d_log(minimum, maximum, steps, allow_interpolation=True):
    log_min = np.log(minimum)
    log_max = np.log(maximum)
    round_decimals = 16
    return {
        'map_from_uniform': lambda u: round(np.exp(u * (log_max - log_min) + log_min), round_decimals),
        'map_to_uniform': lambda x: round((np.log(x) - log_min) / (log_max - log_min), round_decimals),
        'steps': steps,
        'allow_interpolation': allow_interpolation,
    }



def _validate_metaparamspace(metaparamspace):
    for key in metaparamspace:
        mps = metaparamspace[key]
        if type(mps) is not dict:
            raise TypeError(f'Metaparamspace description @key "{key}" must be a dictionary, but was {type(mps)}')
        if 'values' in mps:
            if type(mps['values']) is not list:
                raise ValueError(f'Meta space description @key "{key}": "value" field must be a list, but was type {type(mps["values"])}.')
            if len(mps["values"]) == 0:
                raise ValueError(f'Meta space description @key "{key}": "value" field cannot be empty.')
        if 'steps' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain number of "steps" or a list of "values", got {", ".join(mps.keys())}.')
        if 'map_to_uniform' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain a function "map_to_uniform" or a list of "values", got {mps.keys()}.')
        if 'map_from_uniform' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain a function "map_from_uniform" or a list of "values", got {mps.keys()}.')
    return

def _unroll_1d(metaparamspace1d):
    mps = metaparamspace1d
    _validate_metaparamspace({'_unroll_1d': mps})
    
    if 'values' in mps:
        return mps['values']
    
    is_int = mps.get('integer', False)
    steps = mps['steps']
    map_to_uniform = mps['map_to_uniform']
    map_from_uniform = mps['map_from_uniform']

    value_list = map(map_from_uniform, np.linspace(0, 1, num=steps))
    return value_list
    

def _rm_duplicates(values):
    return list(OrderedDict.fromkeys(values))


# https://arxiv.org/abs/1605.07079
# http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
