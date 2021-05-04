################################################################################
# Diabetes onset by Multi-Layer Perceptron method from scratch in Python       #
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

def softmax(y):
    return np.exp((y)) / np.sum(np.exp(y))

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

def plot_feature_histograms_for_a_target(separated, target, feature_names, target_name, bin_count):
    ds_measured = separated[target]
    
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

def explore_data(ds):
    print(ds.head())
    print(ds.describe())

    print('===== feature histograms per target')
    separated = separate_by_targets(ds.values[:,:-1], ds.values[:,-1])
    for target in np.arange(len(separated.keys())):
        feature_names = ds.columns[:-1]
        target_name = str(target)
        plot_feature_histograms_for_a_target(separated, target, feature_names, target_name, 24)

################################################################################
# Classes                                                                      #
################################################################################

class MLP:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size) / 1000
        self.by = np.zeros((output_size, 1))

    def forward(self, x):
        self.last_x = x

        h = np.tanh(self.Wxh @ x + self.bh)
        self.last_h = h

        y = self.Why @ h + self.by

        return y, h

    def backward(self, d_y, learn_rate=2e-2):
        d_Why = d_y @ self.last_h.T
        d_by = d_y

        d_h = self.Why.T @ d_y

        temp = (1 - self.last_h ** 2) * d_h

        d_Wxh = temp @ self.last_x.T
        d_bh = temp
            
        for d in [d_Wxh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

    def train(self, X_train, y_train, backprop=True):
        loss = 0
        num_correct = 0

        row_count = X_train.shape[0]
        for row in np.arange(row_count):
            measured = X_train[row, :].reshape(-1, 1)
            target = y_train[row]

            y, h = self.forward(measured)

            p = softmax(y)
            loss += -np.log(p[target])
            num_correct += int(np.argmax(p) == target)

            if backprop:
                d_y = p
                d_y[target] -= 1
                
                self.backward(d_y)

        loss = loss[0] / row_count
        acc = num_correct / row_count

        return loss, acc

    def predict(self, measured):
        y, h = self.forward(measured)
        p = softmax(y)

        return np.argmax(p)

################################################################################
# Variables                                                                    #
################################################################################

input_size = 8
hidden_size = 16
output_size = 2
epochs = 500

sample_data = np.array([[6, 148, 72, 35, 100, 50, 1, 40], [1, 148, 72, 35, 100, 50, 1, 40]])

################################################################################
# Main                                                                         #
################################################################################

def main(explore=False):
    np.set_printoptions(precision=6)
    np.random.seed(7)
    
    # 1. Load data
    ds_diabetes = pd.read_csv('datasets_228_482_diabetes.csv')
    
    # Describe data and plot feature histograms per target
    if explore:
        explore_data(ds_diabetes)
    
    # 2. Split into train and test datasets
    X = ds_diabetes.values[:,:-1]
    y = [int(v) for v in ds_diabetes.values[:,-1]]
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Fit with the train dataset
    mlp = MLP(input_size, output_size, hidden_size)

    print('==== Train ====')
    for epoch in range(epochs):
        loss, acc = mlp.train(X_train, y_train, True)
        if epoch % 1 == 0:
            print(f'** Epoch: {epoch + 1} **')
            print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

    # 4. Evaluate with the test dataset
    print('==== Test ====')
    loss, acc = mlp.train(X_test, y_test, False)
    print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

    # 5. Predict with the new input data
    print(f'===== Predict')
    for row in range(sample_data.shape[0]):
        measured = sample_data[row]
        target = mlp.predict(measured.reshape(-1, 1))
        print(f'{measured} => {target}')

if __name__ == '__main__':
    try:
        explore = False
        if len(sys.argv) == 2 and sys.argv[1] == 'e':
            explore = True

        main(explore)
    except:
        traceback.print_exc(file=sys.stdout)
