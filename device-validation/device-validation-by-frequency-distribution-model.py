################################################################################
# Device validation by frequency distribution model                            #
#                                                                              #
# Features:                                                                    #
#   1. Load dataset from file.                                                 #
#   2. Plot scatter diagram of dataset.                                        #
#   3. Plot posterior distribution with a given measured value.                #
################################################################################

import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

################################################################################
# Constants                                                                    #
################################################################################

BINS_ACTUAL = 8
RANGE_ACTUAL = (0, 400)

BINS_MEASURED = 10
RANGE_MEASURED = (0, 100)

################################################################################
# Functions                                                                    #
################################################################################

def create_data(sample_count):
    gold_standard = np.random.normal(200, 50, sample_count)
    
    r = np.random.normal(3, 5, sample_count)
    device_a = 0.3 * gold_standard - 10 + r
    
    r = np.random.normal(3, 5, sample_count)
    device_b = 0.01 * gold_standard + 70 + r
    
    dataset_a = list(zip(device_a, gold_standard))
    dataset_b = list(zip(device_b, gold_standard))
    
    dataframe_a = pd.DataFrame(dataset_a, columns = ['new_device', 'gold_standard'])
    dataframe_b = pd.DataFrame(dataset_b, columns = ['new_device', 'gold_standard'])
    
    return dataframe_a, dataframe_b

def save_data(dataframe, filename):
    dataframe.to_csv(filename, index = None, header = True)
    
def load_data(filename):
    return pd.read_csv(filename)
    
def plot_scatter_diagram(data_actual, data_measured, title):
    plt.figure(figsize = (8, 4))
    plt.scatter(data_actual, data_measured)
    plt.xlim(0, 400)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.xlabel('actual')
    plt.ylabel('measured')
    plt.title(title)
    plt.show()
    
def get_frequency_table(data_measured):
    min = RANGE_ACTUAL[0]
    max = RANGE_ACTUAL[1]
    bin_width = (max - min) / BINS_ACTUAL
    
    frequency_table = {}
    
    for bin_index in np.arange(BINS_ACTUAL):
        frequency_table[bin_index] = []
    
    for i in np.arange(len(data_actual)):
        bin_index = int((data_actual[i] - min) / bin_width)
        frequency_table[bin_index].append(data_measured[i])
        
    return frequency_table

def get_hist_table(frequency_table):
    hist_table = {}
    for bin_index in np.arange(BINS_ACTUAL):
        hist, _ = np.histogram(frequency_table[bin_index], BINS_MEASURED, RANGE_MEASURED)
        hist_table[bin_index] = hist
        
    return hist_table

def get_bin_index(bins, range, value):
    min = range[0]
    max = range[1]
    bin_width = (max - min) / bins
    bin_index = int((value - min) / bin_width)
    return bin_index
    
def get_probability(data, bins, range, value):
    hist, bin_edges = np.histogram(data, bins = bins, range = range)
    bin_index = get_bin_index(bins, range, value)
    probability = hist[bin_index] / len(data)
    return probability

def get_prior(actual):
    prior = get_probability(data_actual, BINS_ACTUAL, RANGE_ACTUAL, actual)
    return prior
    
def get_likelihood(hist_table, actual, measured):
    bin_index_actual = get_bin_index(BINS_ACTUAL, RANGE_ACTUAL, actual)
    bin_index_measured = get_bin_index(BINS_MEASURED, RANGE_MEASURED, measured)
    hist = hist_table[bin_index_actual]
    if np.sum(hist) == 0:
        likelihood = 0
    else:
        likelihood = hist[bin_index_measured] / np.sum(hist)
    return likelihood
    
def get_marginal_likelihood(hist_table, measured):
    total_count = 0
    for bin_index_actual in np.arange(BINS_ACTUAL):
        hist = hist_table[bin_index_actual]
        total_count += np.sum(hist)
    
    bin_index_measured = get_bin_index(BINS_MEASURED, RANGE_MEASURED, measured)
    
    marginal_likelihood = 0
    for bin_index_actual in np.arange(BINS_ACTUAL):
        hist = hist_table[bin_index_actual]
        marginal_likelihood += hist[bin_index_measured] / total_count
        
    return marginal_likelihood

def get_posterior(hist_table, actual, measured):
    prior = get_prior(actual)
    likelihood = get_likelihood(hist_table, actual, measured)
    marginal_likelihood = get_marginal_likelihood(hist_table, measured)
    if marginal_likelihood == 0:
        posterior = 0
    else:
        posterior = likelihood * prior / marginal_likelihood    
    return posterior
    
def plot_posterior_distribution(hist_table, measured, title):
    actual_arr = np.linspace(RANGE_ACTUAL[0], RANGE_ACTUAL[1], 400, endpoint = False)
    posterior_arr = []

    for actual in actual_arr:
        posterior = get_posterior(hist_table, actual, measured)
        posterior_arr.append(posterior)

    plt.figure(figsize = (8, 4))

    plt.plot(actual_arr, posterior_arr)
    plt.grid(True)
    plt.xlim(RANGE_ACTUAL[0], RANGE_ACTUAL[1])
    plt.ylim(0, 1.0)
    plt.xlabel('actual')
    plt.ylabel('probability')
    plt.title(title)

    plt.show()

def print_usage(script_name):
  print(f'Usage: python {script_name} <dataset_file> <measured_value>')
  
################################################################################
# Main                                                                         #
################################################################################

if len(sys.argv) < 3:
  print_usage(sys.argv[0])
  sys.exit(-1)

dataset_file = sys.argv[1]
measured_a = int(sys.argv[2])

np.random.seed(7)

dataframe_a = load_data(dataset_file)

data_actual = dataframe_a['gold_standard']
data_measured_a = dataframe_a['new_device']

correlation_a, _ = pearsonr(data_measured_a, data_actual)
print(f'Correlation between DEVICE and GOLD_STANDARD = {correlation_a:.2f}')

plot_scatter_diagram(data_actual, data_measured_a, f'dataset (corr = {correlation_a:.2f})')

frequency_table_a = get_frequency_table(data_measured_a)
hist_table_a = get_hist_table(frequency_table_a)

title_a = f'posterior distribution with measured = {measured_a}'
plot_posterior_distribution(hist_table_a, measured_a, title_a)
