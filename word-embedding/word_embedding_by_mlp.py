################################################################################
# Word Embedding by Multi-Layer Perceptron method from scratch in Python       #
################################################################################

import os, sys, traceback
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

################################################################################
# Functions                                                                    #
################################################################################

def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    word_to_id = {}
    id_to_word = {}
    
    token_set = list(dict.fromkeys(tokens))
    
    for i, token in enumerate(token_set):
        word_to_id[token] = i
        id_to_word[i] = token
    
    return word_to_id, id_to_word

def softmax(y):
    return np.exp((y)) / np.sum(np.exp(y))

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
    
    return np.asarray(X), np.asarray(y)

################################################################################
# Classes                                                                      #
################################################################################

class MLP:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Wxh = np.random.randn(hidden_size, input_size)
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size)
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
        #self.bh -= learn_rate * d_bh
        #self.by -= learn_rate * d_by

    def train(self, X_train, y_train, backprop=True, learn_rate=2e-2):
        loss = 0
        num_correct = 0
        
        row_count = X_train.shape[0]
        for row in np.arange(row_count):
            measured = X_train[row, :].reshape(-1, 1)
            target = np.argmax(y_train[row])

            y, h = self.forward(measured)

            p = softmax(y)
            loss += -np.log(p[target])
            num_correct += int(np.argmax(p) == target)

            if backprop:
                d_y = p
                d_y[target] -= 1
                
                self.backward(d_y, learn_rate)

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

epochs = 50

sample_data = ['static', 'string']

# Excerpt from https://dart.dev/guides/language/type-system
text_data = 'Soundness is about ensuring your program can’t get into certain invalid states. \
    A sound type system means you can never get into a state where an expression evaluates \
    to a value that doesn’t match the expression’s static type. For example, if an expression’s \
    static type is String, at runtime you are guaranteed to only get a string when you evaluate it. \
    Dart’s type system, like the type systems in Java and C#, is sound. It enforces that soundness \
    using a combination of static checking (compile-time errors) and runtime checks. For example, \
    assigning a String to int is a compile-time error. Casting an object to a String using as \
    String fails with a runtime error if the object isn’t a String.'

################################################################################
# Main                                                                         #
################################################################################

def main(filepath, plot=False):
    np.set_printoptions(precision=6)
    np.random.seed(42)
    
    # 1. Load data
    if filepath is not None:
        with open(filepath, mode='r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = text_data

    tokens = tokenize(text)
    word_to_id, id_to_word = mapping(tokens)
    print(id_to_word)
    
    window = 2
    X, y = generate_training_data(tokens, word_to_id, window)

    # 2. Split into train and test datasets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train, X_test, y_test = X, y, X, y

    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_train[0]: {X_train[0]}')
    print(f'np.argmax(X_train[0]): {np.argmax(X_train[0])} ({id_to_word[np.argmax(X_train[0])]})')
    
    # 3. Fit with the train dataset
    input_size = len(word_to_id)
    hidden_size = 10
    output_size = len(word_to_id)
    learn_rate = 0.05

    mlp = MLP(input_size, output_size, hidden_size)

    print('==== Train ====')
    history = []
    for epoch in range(epochs):
        loss, acc = mlp.train(X_train, y_train, True, learn_rate)
        history.append(loss)
        if epoch % 1 == 0:
            print(f'Epoch: {epoch + 1:4d}, loss: {loss:.4f}, accuracy: {acc:.4f}')

    if plot:
        plt.style.use("seaborn")
        plt.plot(range(len(history)), history, color="skyblue")
        plt.show()

    # 4. Evaluate with the test dataset
    print('==== Test ====')
    loss, acc = mlp.train(X_test, y_test, False)
    print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

    # 5. Predict with the new input data
    print(f'===== Predict')
    for row in range(len(sample_data)):
        input_word = sample_data[row]
        measured = np.array(one_hot_encode(word_to_id[input_word], len(word_to_id)))
        target = mlp.predict(measured.reshape(-1, 1))
        print(f'{input_word} => {id_to_word[target]}')

if __name__ == '__main__':
    try:
        filepath = None
        if len(sys.argv) >= 3 and 'f' in sys.argv[1]:
            filepath = sys.argv[2]

        plot = False
        if len(sys.argv) >= 2 and 'p' in sys.argv[1]:
            plot = True

        main(filepath, plot)
    except:
        traceback.print_exc(file=sys.stdout)
