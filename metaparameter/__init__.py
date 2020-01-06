import numpy as np
from itertools import product
import pandas as pd
from collections import OrderedDict

def random(metaparamspace):
    """selects metaparams randomly"""
    params = {}
    for key in metaparamspace:
        value_list = list(unroll_1d(metaparamspace[key]))
        params[key] = value_list[np.random.choice(len(value_list))]
    return params

def grid_search(metaparamspace, experience, density=1.0):
    """selects randomly metaparams, with the least epochs trained with"""
    keys, grid = unroll(metaparamspace)
    grid_epochs = []
    for p in grid:
        experience_run = experience
        for i in range(len(keys)):
            key = keys[i]
            if key in experience_run.columns:
                value = p[i]
                experience_run = experience_run[experience_run[key]==value]
            else: experience_run = pd.DataFrame()
        epochs = len(experience_run)
        grid_epochs.append((p, epochs))
    epoch_min = min(map(lambda x: x[1], grid_epochs))
    grid_min = filter(lambda x: x[1] == epoch_min, grid_epochs)
    grid_min_params = list(map(lambda x: x[0], grid_min))
    params = grid_min_params[np.random.choice(len(grid_min_params))]
    return dict(zip(keys, params))

def unroll(metaparamspace):
    grid_space = {}
    for key in metaparamspace:
        grid_space[key] = unroll_1d(metaparamspace[key])
    # grid_space_keys = sorted(grid_space.keys())
    # grid_space_values = [grid_space[key] for key in grid_space_keys]
    # grid = list(product(*grid_space_values))
    # return grid_space_keys, grid
    return grid_space


def unroll_1d(metaparamspace1d):
    mps = metaparamspace1d
    if type(mps) == list:
        value_list = mps
    elif type(mps) == dict:
        is_int = mps.get('integer', False)
        transform_from_uniform = mps.get('transform_from_uniform', lambda x: x)
        transform_to_uniform = mps.get('transform_to_uniform', lambda x: x)
        min_uniform = transform_to_uniform(mps['min'] if not is_int else np.ceil(mps['min']))
        max_uniform = transform_to_uniform(mps['max'] if not is_int else np.floor(mps['max']))
        steps = mps['steps']
        space_uniform = np.linspace(min_uniform, max_uniform, num=steps)
        value_list = map(transform_from_uniform, space_uniform)
        if is_int:
            # round numbers, clamp to [min, max], cast to int, remove duplicates
            value_list = _rm_duplicates(map(int,
                                            map(lambda v: np.clip(v, mps['min'], mps['max']),
                                                map(np.round,
                                                    value_list))))
    return value_list


def _rm_duplicates(values):
    return list(OrderedDict.fromkeys(values))


def space1d_linear(minimum, maximum, steps, integer=False):
    return {
        'min': minimum,
        'max': maximum,
        'steps': steps,
        'integer': integer,
    }

def space1d_log10(minimum, maximum, steps, integer=False):
    return {
        'min': minimum,
        'max': maximum,
        'steps': steps,
        'transform_to_uniform': np.log10,
        'transform_from_uniform': lambda x: np.power(10, x),
        'integer': integer,
    }


# https://arxiv.org/abs/1605.07079
# http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
