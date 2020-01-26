import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import MLenv.metaparameter
from . import space
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, PairwiseKernel, Product, WhiteKernel

PER_SAMPLE_VARIANCE = 0.01 # relative to metric variance

def fixed_invest_gp(metaparamspace, experience, objective_column, invest_value, invest_type):
    # select only relevant columns
    columns = {objective_column, invest_type}
    columns.update(list(metaparamspace.keys()))
    experience = experience.get(columns, pd.DataFrame())
    
    # if there is no relevant experience, choose random metaparameters
    if len(experience) == 0:
        return MLenv.metaparameter.random(metaparamspace)

    # filter experience for epoch/time >= invest_value
    experience = experience[experience[invest_type] >= invest_value]

    # select the row with smallest invest_value for each choice of metaparameters
    experience = experience.sort_values(invest_type)
    experience = experience.groupby(list(metaparamspace.keys())).first()
    experience = experience.reset_index()

    # select nominal parameters e.g. with random searcher
    nominal_columns = list(filter(lambda p: 'map_to_uniform' not in metaparamspace[p], metaparamspace.keys()))
    metaparamspace_nominals = {key_nominal: metaparamspace[key_nominal] for key_nominal in nominal_columns}
    metaparams = MLenv.metaparameter.random(metaparamspace_nominals)

    # filter experience according to selected nominals
    for key in nominal_columns:
        experience = experience[experience[key] == metaparams[key]]

    kardinal_columns = list(filter(lambda c: c not in nominal_columns, metaparamspace.keys()))

    # transform into uniform space
    for key in metaparamspace:
        if key in kardinal_columns and key in experience.columns:
            experience[key] = experience[key].apply(metaparamspace[key]['map_to_uniform'])
    

    # get the data for estimating the best choice of metaparameters
    X = experience[kardinal_columns].values.tolist()
    Y = experience[objective_column].values.tolist()

    assert(len(X) == len(Y))

    # choose the first two meassurements at random
    if len(X) < 2:
        return MLenv.metaparameter.random(metaparamspace)

    # preparing the gaussian process

    # how smooth the curve will be.
    # choosing interval length over sample count as a fixed value.
    length_scales = 1 / np.array(experience[kardinal_columns].nunique())
    length_scale_bounds = [(ls, ls) for ls in length_scales]

    # scale the kernel with data variance (or possibly a constant times that),
    # so that results are invariant under scaling the metric.
    var = np.std(Y)**2

    # set the per sample variance (diagonals of the kernel) dependent of the data variance, again for scaling invariance
    whls = var * PER_SAMPLE_VARIANCE

    # build up a gaussian process
    kernel = RBF(length_scales, length_scale_bounds) * C(var, (var, var)) + (0 if whls<=0 else WhiteKernel(whls, (whls, whls)))
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # fit the gaussian process
    gp.fit(X, Y)
    
    # evaluate expected improvement. This measures the probability,
    # that a run with some metaparameters will be at least somewhat better as a reference value

    # The reference used is the maximum value plus one standard deviation over the data values
    reference = np.max(Y) + np.std(Y)

    # evaluate the gp at n points and compare /find the best expected improvement
    n = 200

    # the n sample points are sampled according to the metaparamspace definition,
    # also again mapped to uniform space
    samples = np.zeros((n, len((kardinal_columns))))
    samples_uniform = np.zeros((n, len((kardinal_columns))))
    for i, key in enumerate(kardinal_columns):
        samples[:,i] = space.sample_1d(metaparamspace[key], n)
        samples_uniform[:,i] = metaparamspace[key]['map_to_uniform'](samples[:,i])

    # gp predicts a gaussian over the metric for each sample
    mean, sigma = gp.predict(samples_uniform, return_std=True)

    # the required probability is 1-cdf((r-m)/s)), however cdf is strictly monotonic and has bad numerical properties, and therefor ommited.
    expected_improvement = (np.array([(reference - m) / s for m, s in zip(mean, sigma)]))
    # Also the "1 - " was ommitted, instead the minumal value is now the optimal chioce
    gp_choice_index = np.argmin(expected_improvement)
    gp_choice = samples_uniform[gp_choice_index]

    # find the correct values in parameter domain
    for i, key in enumerate(kardinal_columns):
        metaparams[key] = samples[gp_choice_index, i]
    
    # every metaparameter should be chosen by now
    assert(metaparams.keys() == metaparamspace.keys())

    # this method can only visualize what's going on, if there is just one kardinal parameter.
    # _plot(gp, X, Y, samples_uniform, mean, sigma)

    return metaparams



def _plot(gp, X, Y, sampled_X, sampled_Y, sampled_sigma):
    x_margin = 0

    XX = np.atleast_2d(np.linspace(0-x_margin, 1+x_margin, 1000)).T

    YY, sigma = gp.predict(XX, return_std=True)

    reference = np.max(Y) + np.std(Y)
    ex_imp = (np.array([(reference - m) / s for m, s in zip(YY, sigma)]))
    sampled_ex_imp = (np.array([(reference - m) / s for m, s in zip(sampled_Y, sampled_sigma)]))

    plt.figure()
    plt.xlim(0-x_margin, 1+x_margin)
    plt.plot(XX, ex_imp, 'b-', label='exp. improvement')
    plt.plot(sampled_X, sampled_ex_imp, 'g.', markersize=10, label='sampled exp. improvement')

    plt.figure()
    plt.xlim(0-x_margin, 1+x_margin)
    plt.plot(sampled_X, sampled_Y, 'g.', markersize=10, label='Observations')
    plt.plot(XX, YY, 'b-', label='Prediction')
    plt.fill(np.concatenate([XX, XX[::-1]]),
            np.concatenate([YY - 1.9600 * sigma,
                            (YY + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')

    plt.fill(np.concatenate([XX, XX[::-1]]),
            np.concatenate([YY - 1 * sigma,
                            (YY + 1 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='sigma interval')
    plt.plot(X, Y, 'r.', markersize=10, label='Observations')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='lower center')

    print(plt.get_backend())

    plt.show()