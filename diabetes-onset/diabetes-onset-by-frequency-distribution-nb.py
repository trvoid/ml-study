################################################################################
# Predict the onset of diabetes by frequency distribution naive bayes method   #
# from scratch in Python                                                       #
################################################################################

import os, sys, traceback
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

################################################################################
# Functions                                                                    #
################################################################################

def load_data():
    return pd.read_csv('datasets_228_482_diabetes.csv')

def separate_by_target(X, y):
    separated_dict = defaultdict(lambda: [])
    
    row_count = X.shape[0]
    for row in np.arange(row_count):
        measured = X[row, :]
        target = y[row]
        separated_dict[target].append(measured)
    
    for target in separated_dict.keys():
        separated_dict[target] = np.array(separated_dict[target])
        
    return separated_dict

def plot_feature_histograms_for_a_target(ds_measured, feature_names, target_name, bin_count):
    fig = plt.figure(figsize = (16,4))
    fig.suptitle(f'feature histograms for a target: {target_name}')
    for col in np.arange(len(feature_names)):
        plt.subplot(241 + col)
        plt.hist(ds_measured[:, col], bins=bin_count)
        plt.grid(True)
        plt.xlabel('measured')
        plt.ylabel('frequency')
        plt.title(feature_names[col])
    
    plt.show()

def explore_data(ds, bin_count):
    print(ds.head())
    print(ds.describe())

    print('===== feature histograms per target')
    separated_dict = separate_by_target(ds.values[:,:-1], ds.values[:,-1])
    for target in np.arange(len(separated_dict.keys())):
        feature_names = ds.columns[:-1]
        target_name = f'{ds.columns[-1]}({target})'
        plot_feature_histograms_for_a_target(separated_dict[target], feature_names, target_name, bin_count)

def make_bin_edges(data, bin_count):
    bin_edges_dict = {}
    
    for col in np.arange(data.shape[1]):
        _, bin_edges = np.histogram(data[:, col], bins=bin_count)
        #print(f'col {col}: {bin_edges}')
        bin_edges_dict[col] = bin_edges

    return bin_edges_dict

def get_bin_index(bin_edges, value):
    for i in np.arange(len(bin_edges) - 1):
        if value >= bin_edges[i] and value < bin_edges[i+1]:
            return i

    return i

################################################################################
# Classes                                                                      #
################################################################################

class FrequencyNB:
    def __init__(self, bin_edges_list):
        self.bin_edges_list = bin_edges_list

    def fit(self, X, y):
        self.priors = {}

        separated = separate_by_target(X, y)

        for target in separated.keys():
            self.priors[target] = separated[target].shape[0] / X.shape[0]

        self.p_dist_dict = defaultdict(lambda: [])

        for target in separated.keys():
            for col in np.arange(X.shape[1]):
                hist, _ = np.histogram(separated[target][:,col], bins=self.bin_edges_list[col])
                #print(f'col {col}: {hist}')
                p_dist = hist / np.sum(hist)
                self.p_dist_dict[target].append(p_dist)

    def predict(self, X):
        result = np.zeros(X.shape[0])

        for row in np.arange(X.shape[0]):
            input = X[row, :]
            posteriors = {}

            for target in self.priors.keys():
                prior = self.priors[target]
                p_dist = self.p_dist_dict[target]

                likelihood = 1.0

                for col in np.arange(len(input)):
                    bin_index = get_bin_index(self.bin_edges_list[col], input[col])
                    likelihood *= p_dist[col][bin_index]

                posteriors[target] = likelihood * prior

            #print(posteriors)

            max_posterior = -1
            max_target = -1
            for target in posteriors.keys():
                if posteriors[target] > max_posterior:
                    max_posterior = posteriors[target]
                    max_target = target

            result[row] = max_target

        return result

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

################################################################################
# Variables                                                                    #
################################################################################

explore_bin_count = 10
fit_bin_count = 5

sample_data = np.array([[6, 148, 72, 35, 100, 50, 1, 40], [1, 148, 72, 35, 100, 50, 1, 40]])

################################################################################
# Main                                                                         #
################################################################################

def main(explore=False):
    np.set_printoptions(precision=6)
    np.random.seed(7)
    
    # 1. Load data
    ds = load_data()

    # Explore data
    if explore:
        explore_data(ds, explore_bin_count)

    X = ds.values[:,:-1]
    y = [int(v) for v in ds.values[:,-1]]
    
    # 2. Make bin edges
    bin_edges_dict = make_bin_edges(X, fit_bin_count)

    # 3. Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. Fit the classifier
    nb_classifier = FrequencyNB(bin_edges_dict)
    nb_classifier.fit(X_train, y_train)

    # 5. Score the performance
    score = nb_classifier.score(X_test, y_test)
    print(f'score: {score:.4f}')

    # 6. Predict on sample data
    y_pred = nb_classifier.predict(sample_data)
    print(y_pred)

if __name__ == '__main__':
    try:
        explore = False
        if len(sys.argv) == 2 and sys.argv[1] == 'e':
            explore = True

        main(explore)
    except:
        traceback.print_exc(file=sys.stdout)
