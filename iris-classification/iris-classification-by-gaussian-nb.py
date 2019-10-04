################################################################################
# Iris classification by Gaussian Naive Bayes method from scratch in Python    #
################################################################################

import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    target_count = thetas.shape[0]
    instance_count = measured.shape[0]

    likelihoods = np.zeros((instance_count, target_count))
    
    for target in np.arange(target_count):
        l = norm.pdf(measured, thetas[target, :], sigmas[target, :])
        likelihoods[:, target] = np.prod(l, axis=1)
        
    return likelihoods
    
def get_posteriors(priors, thetas, sigmas, X):
    likelihoods = get_likelihoods(thetas, sigmas, X)
    marginal_likelihoods = np.sum(likelihoods * priors, axis=1)
    likelihood_ratios = likelihoods / marginal_likelihoods.reshape(len(marginal_likelihoods), -1)
    posteriors = likelihood_ratios * priors
    return posteriors
    
def explore_data(ds_iris):
    print('===== feature names')
    print(ds_iris.feature_names)
    
    print('===== data description')
    df_data = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
    print(df_data.describe())

    print('===== feature histograms per target')
    separated = separate_by_targets(ds_iris.data, ds_iris.target)
    for target in np.arange(len(separated.keys())):
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
        posteriors = get_posteriors(self.priors, self.thetas, self.sigmas, X)
        predicted = np.argmax(posteriors, axis=1)
        return predicted
        
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
        
################################################################################
# Main                                                                         #
################################################################################
    
def main(explore=False):
    np.set_printoptions(precision=6)
    np.random.seed(7)
    
    # 1. Load data
    ds_iris = load_iris()

    print('===== target names')
    print(ds_iris.target_names)

    # Describe data and plot feature histograms per target
    if explore:
        explore_data(ds_iris)
    
    # 2. Split into train and test datasets
    X, y = ds_iris.data, ds_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Fit with the train dataset by Gaussian Naive Bayes method
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # 4. Evaluate with the test dataset
    print(f'===== score')
    score = nb.score(X_test, y_test)
    print(f'{score:.4f}')

    # 5. Predict with the new input data
    measured = np.array([[5.1, 3.3, 1.4, 0.2], [6.1, 3.3, 5.1, 2.4]])
    targets = nb.predict(measured)
    print(f'===== predicted')
    for i in np.arange(measured.shape[0]):
        print(f'{measured[i,:]} => {ds_iris.target_names[targets[i]]}')
    
if __name__ == '__main__':
    explore = False
    if len(sys.argv) == 2 and sys.argv[1] == 'e':
        explore = True

    main(explore)
