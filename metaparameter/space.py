import numpy as np
from itertools import product
import pandas as pd
from collections import OrderedDict



def validate_metaparamspace(metaparamspace):
    for key in metaparamspace:
        mps = metaparamspace[key]
        if type(mps) is not dict:
            raise TypeError(f'Metaparamspace description @key "{key}" must be a dictionary, but was {type(mps)}')
        if 'values' in mps:
            if type(mps['values']) is not list:
                raise ValueError(f'Meta space description @key "{key}": "value" field must be a list, but was type {type(mps["values"])}.')
            if len(mps["values"]) == 0:
                raise ValueError(f'Meta space description @key "{key}": "value" field cannot be empty.')
            return
        if 'steps' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain number of "steps" or a list of "values", got {", ".join(mps.keys())}.')
        if 'map_to_uniform' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain a function "map_to_uniform" or a list of "values", got {mps.keys()}.')
        if 'map_from_uniform' not in mps:
            raise ValueError(f'Meta space description @key "{key}" must have either contain a function "map_from_uniform" or a list of "values", got {mps.keys()}.')
    return



def nominal(values):
    space = {
        'values': values,
    }
    return space

def linear(minimum, maximum, steps=None, allow_interpolation=True):
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

def log(minimum, maximum, steps, allow_interpolation=True):
    log_min = np.log(minimum)
    log_max = np.log(maximum)
    round_decimals = 16
    return {
        'map_from_uniform': lambda u: np.round(np.exp(u * (log_max - log_min) + log_min), round_decimals),
        'map_to_uniform': lambda x: np.round((np.log(x) - log_min) / (log_max - log_min), round_decimals),
        'steps': steps,
        'allow_interpolation': allow_interpolation,
    }



def unroll_1d(metaparamspace1d):
    """
    List all values of a metaspace dimension according to the distribution described in the metaparamspace description.
    Values will be linearly distributed in uniform space, and "steps" is used for that.
    "allow_interpolation" is ignored by this method.
    """
    mps = metaparamspace1d
    validate_metaparamspace({'_unroll_1d': mps})
    
    if 'values' in mps:
        return mps['values']
    
    is_int = mps.get('integer', False)
    steps = mps['steps']
    map_to_uniform = mps['map_to_uniform']
    map_from_uniform = mps['map_from_uniform']

    value_list = map(map_from_uniform, np.linspace(0, 1, num=steps))
    if is_int:
        value_list = np.round(value_list).astype(np.int)
    return value_list
    
def sample_1d(metaparamspace1d, n):
    """
    sample from metaspace dimension according to the distribution described in the metaparamspace description
    """
    mps = metaparamspace1d
    validate_metaparamspace({'_unroll_1d': mps})

    is_int = mps.get('integer', False)

    if 'values' in mps:
        values = np.array(mps['values'])
        if is_int:
            values = np.round(values).astype(np.int)
        return values[np.random.choice(len(values), n)]
    

    allow_interpolation = mps.get('allow_interpolation', False)
    steps = mps['steps']
    map_to_uniform = mps['map_to_uniform']
    map_from_uniform = mps['map_from_uniform']

    # can the values be interpolated or are only values according to step size valid?
    if allow_interpolation:
        samples_uniform = np.random.uniform(0, 1, size=n)
        samples = map_from_uniform(samples_uniform)
        if is_int:
            samples = np.round(samples).astype(np.int)
        return samples
    else:
        values_uniform = np.linspace(0, 1, num=steps)
        samples_uniform = values_uniform[np.random.choice(len(values_uniform), n)]
        samples = map_from_uniform(samples_uniform)
        if is_int:
            samples = np.round(samples).astype(np.int)
        return samples


def snap(metaparams, experience, threshold=0.5, exclude_experience=None):
    # map metaparams to existing ones, if they are in each uniform dimension closer than threshold/steps
    return metaparams



