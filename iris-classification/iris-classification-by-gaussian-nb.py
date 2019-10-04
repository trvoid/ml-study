################################################################################
# Iris classification by Gaussian Naive Bayes method from scratch in Python    #
################################################################################

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
    targets = separated.keys()
    target_count = len(targets)
    feature_count = separated[0].shape[1]

    thetas = np.zeros((target_count, feature_count))
    sigmas = np.zeros((target_count, feature_count))
    
    for target in targets:
        ds_measured = separated[target]
        for col in np.arange(feature_count):
            theta = np.mean(ds_measured[:, col])
            sigma = np.std(ds_measured[:, col])
            thetas[target, col] = theta
            sigmas[target, col] = sigma
        
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
    targets = separated.keys()

    priors = np.zeros(len(targets))
    
    total_count = 0
    for target in targets:
        count = separated[target].shape[0]
        total_count += count
        priors[target] = count
    
    priors /= total_count
    
    return priors

def get_likelihoods(thetas, sigmas, measured):
    likelihoods = {}
    
    for target in np.arange(len(thetas)):
        likelihood = 1.0
        for col in np.arange(len(measured)):
            theta = thetas[target, col]
            sigma = sigmas[target, col]
            l = norm.pdf(measured[col], theta, sigma)
            likelihood *= l
        likelihoods[target] = likelihood
        
    return likelihoods
    
def get_posteriors(priors, thetas, sigmas, X):
    # get likelihoods
    likelihoods = get_likelihoods(thetas, sigmas, X)
    
    # get marginal likelihood
    marginal_likelihood = 0.0
    for target in np.arange(len(priors)):
        prior = priors[target]
        likelihood = likelihoods[target]
        marginal_likelihood += likelihood * prior
    
    # get posteriors
    posteriors = {}
    for target in np.arange(len(priors)):
        prior = priors[target]
        likelihood = likelihoods[target]
        posterior = likelihood * prior / marginal_likelihood
        posteriors[target] = posterior
    
    return posteriors
    
def explore_data(ds_iris):
    print('===== data description')
    df_data = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
    print(df_data.describe())

    separated = separate_by_targets(ds_iris.data, ds_iris.target)
    for target in separated.keys():
        feature_names = ds_iris.feature_names
        target_name = ds_iris.target_names[target]
        plot_feature_histograms_for_a_target(separated, target, feature_names, target_name)

################################################################################
# Classes                                                                      #
################################################################################

class GaussianNB:
    def fit(self, X, y):
        # separate by targets
        separated = separate_by_targets(X, y)
        
        # get priors
        self.priors = get_priors(separated)
        
        # get norm parameters
        self.thetas, self.sigmas = get_norm_params(separated)
        
    def predict(self, X):
        predicted = []
        
        for row in np.arange(X.shape[0]):
            posteriors = get_posteriors(self.priors, self.thetas, self.sigmas, X[row,:])

            # find a maximum posterior
            max = 0.0
            map = None
            for target in posteriors.keys():
                posterior = posteriors[target]
                if posterior > max:
                    max = posterior
                    map = target
            
            predicted.append(map)
            
        return predicted
        
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
        
################################################################################
# Main                                                                         #
################################################################################
    
def main():
    np.set_printoptions(precision=6)
    np.random.seed(7)
    
    ds_iris = load_iris()

    # Describe data and plot feature histograms per target
    # explore_data(ds_iris)
    
    # Split into train and test datasets
    X, y = ds_iris.data, ds_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit with Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    print(f'===== score')
    score = nb.score(X_test, y_test)
    print(f'{score:.4f}')
            
    measured = np.array([[5.1, 3.3, 1.4, 0.2]])
    y = nb.predict(measured)
    print(f'===== predicted')
    print(y)
    
    

if __name__ == '__main__':
    main()
