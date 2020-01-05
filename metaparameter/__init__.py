import numpy as np
from itertools import product
import pandas as pd
# from collections import OrderedDict

def random(metaparamspace):
    """selects metaparams randomly"""
    params = {}
    for key in metaparamspace:
        mps = metaparamspace[key]
        if type(mps) == list:
            value = np.random.choice(mps)
        elif type(mps) == dict:
            transform_from_uniform = mps.get('transform_from_uniform', lambda x: x)
            transform_to_uniform = mps.get('transform_to_uniform', lambda x: x)
            min_uniform = transform_to_uniform(mps['min'])
            max_uniform = transform_to_uniform(mps['max'])
            steps = mps['steps']
            space_uniform = np.linspace(min_uniform, max_uniform, num=steps)
            value_uniform = np.random.choice(space_uniform)
            value = transform_from_uniform(value_uniform)
        params[key] = value
    return params

def grid_search(metaparamspace, experience, density=1.0):
    """selects randomly metaparams, with the least epochs trained with"""
    keys, grid = _get_full_grid(metaparamspace)
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
    return params

def _get_full_grid(metaparamspace):
    grid_space = {}
    for key in metaparamspace:
        mps = metaparamspace[key]
        if type(mps) == list:
            value_list = mps
        elif type(mps) == dict:
            transform_from_uniform = mps.get('transform_from_uniform', lambda x: x)
            transform_to_uniform = mps.get('transform_to_uniform', lambda x: x)
            min_uniform = transform_to_uniform(mps['min'])
            max_uniform = transform_to_uniform(mps['max'])
            steps = mps['steps']
            space_uniform = np.linspace(min_uniform, max_uniform, num=steps)
            value_list = list(map(transform_from_uniform, space_uniform))
        grid_space[key] = value_list
    grid_space_keys = sorted(grid_space.keys())
    grid_space_values = [grid_space[key] for key in grid_space_keys]
    grid = list(product(*grid_space_values))
    return grid_space_keys, grid

# def _rm_duplicates(values):
#     return list(OrderedDict.fromkeys(t))


# https://arxiv.org/abs/1605.07079
# http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
