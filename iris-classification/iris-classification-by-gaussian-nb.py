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
    
def get_posteriors(priors, thetas, sigmas, X):
    # get likelihoods
    likelihoods = get_likelihoods(thetas, sigmas, X)
    
    # get marginal likelihood
    marginal_likelihood = 0.0
    for target in priors.keys():
        prior = priors[target]
        likelihood = likelihoods[target]
        marginal_likelihood += likelihood * prior
    
    # get posteriors
    posteriors = {}
    for target in priors.keys():
        prior = priors[target]
        likelihood = likelihoods[target]
        posterior = likelihood * prior / marginal_likelihood
        posteriors[target] = posterior
    
    return posteriors
    
################################################################################
# Classes                                                                      #
################################################################################

class GaussianNB:
    def fit(self, X, y):
        # separate by targets
        self.separated = separate_by_targets(X, y)
        
        # get priors
        self.priors = get_priors(self.separated)
        
        # get norm parameters
        self.thetas, self.sigmas = get_norm_params(self.separated)
        
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

    print('===== data description')
    df_data = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
    print(df_data.describe())
    
    # Split into train and test datasets
    X, y = ds_iris.data, ds_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit with Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # plot feature histograms per target
    if False:
        feature_names = ds_iris.feature_names
        for target in nb.separated.keys():
            target_name = ds_iris.target_names[target]
            plot_feature_histograms_for_a_target(nb.separated, target, feature_names, target_name)
    
    print(f'===== score')
    score = nb.score(X_test, y_test)
    print(f'{score:.4f}')
            
    measured = np.array([[5.1, 3.3, 1.4, 0.2]])
    y = nb.predict(measured)
    print(f'===== predicted')
    print(y)
    
    

if __name__ == '__main__':
    main()
