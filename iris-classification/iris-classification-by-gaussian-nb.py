################################################################################
# Iris classification by Gaussian Naive Bayes method from scratch in Python    #
################################################################################

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import norm
import matplotlib.pyplot as plt

################################################################################
# Constants                                                                    #
################################################################################

################################################################################
# Functions                                                                    #
################################################################################

def separate_by_targets(X, y):
    separated = defaultdict(lambda: [])
    
    row_count = X.shape[0]
    for row in np.arange(row_count):
        measured = X[row, :]
        target = y[row]
        separated[target].append(measured)
    
    for target in separated.keys():
        separated[target] = np.array(separated[target])
        
    return separated
    
def get_norm_params(separated):
    thetas = defaultdict(lambda: [])
    sigmas = defaultdict(lambda: [])
    
    for target in separated.keys():
        ds_measured = separated[target]
        for col in np.arange(ds_measured.shape[1]):
            theta = np.mean(ds_measured[:, col])
            sigma = np.std(ds_measured[:, col])
            thetas[target].append(theta)
            sigmas[target].append(sigma)
        
    return thetas, sigmas
    
def plot_feature_histograms_for_a_target(separated, target, feature_names, target_name):
    ds_measured = separated[target]
    
    fig = plt.figure(figsize = (16,4))
    fig.suptitle(f'feature histograms for a target: {target_name}')
    for col in np.arange(len(feature_names)):
        plt.subplot(141 + col)
        plt.hist(ds_measured[:, col], bins=16)
        plt.grid(True)
        plt.xlabel('measured')
        plt.ylabel('frequency')
        plt.title(feature_names[col])
    
    plt.show()
    
def get_priors(separated):
    priors = {}
    
    total_count = 0
    for target in separated.keys():
        count = separated[target].shape[0]
        total_count += count
        priors[target] = count
    
    for target in priors.keys():
        priors[target] = priors[target] / total_count
    
    return priors

def get_likelihoods(thetas, sigmas, measured):
    likelihoods = {}
    
    for target in thetas.keys():
        likelihood = 1.0
        for col in np.arange(len(measured)):
            theta = thetas[target][col]
            sigma = sigmas[target][col]
            l = norm.pdf(measured[col], theta, sigma)
            likelihood *= l
        likelihoods[target] = likelihood
        
    return likelihoods
     
################################################################################
# Main                                                                         #
################################################################################
    
def main():
    np.set_printoptions(precision=6)
    
    ds_iris = load_iris()
    
    print('===== data description')
    df_data = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
    print(df_data.describe())
    
    separated = separate_by_targets(ds_iris.data, ds_iris.target)
    priors = get_priors(separated)
    thetas, sigmas = get_norm_params(separated)
    
    print('===== priors')
    for target in priors.keys():
        print(f'{target}: {priors[target]:.10f}')
    
    # plot feature histograms per target
    if False:
        feature_names = ds_iris.feature_names
        for target in separated.keys():
            target_name = ds_iris.target_names[target]
            plot_feature_histograms_for_a_target(separated, target, feature_names, target_name)
    
    # get likelihoods
    measured = [5.1, 3.3, 1.4, 0.2]
    likelihoods = get_likelihoods(thetas, sigmas, measured)
    print('===== likelihoods')
    for target in likelihoods.keys():
        print(f'{target}: {likelihoods[target]:.10f}')
    
    # get marginal likelihood
    marginal_likelihood = 0.0
    for target in separated.keys():
        prior = priors[target]
        likelihood = likelihoods[target]
        marginal_likelihood += likelihood * prior
    print('===== marginal likelihood')
    print(f'{marginal_likelihood:.10f}')
    
    # get posteriors
    print('===== posteriors')
    posteriors = {}
    for target in separated.keys():
        prior = priors[target]
        likelihood = likelihoods[target]
        posterior = likelihood * prior / marginal_likelihood
        posteriors[target] = posterior
        print(f'{target}: {posterior:.10f}')

    # find a maximum posterior
    max = 0.0
    map = None
    for target in posteriors.keys():
        posterior = posteriors[target]
        if posterior > max:
            max = posterior
            map = target
    
    print(f'===== MAP')
    print(f'MAP: {map} ({ds_iris.target_names[map]})')
    
if __name__ == '__main__':
    main()
