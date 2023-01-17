################################################################################
# Word Embedding by Multi-Layer Perceptron method from scratch in Python       #
################################################################################

import os, sys, traceback
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from konlpy.tag import Hannanum, Kkma, Komoran, Mecab, Okt
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

################################################################################
# Functions                                                                    #
################################################################################

def tokenize(tokenizer_name, text):
    if tokenizer_name == 'Hannanum':
        tokenizer = Hannanum()
    elif tokenizer_name == 'Kkma':
        tokenizer = Kkma()
    elif tokenizer_name == 'Komoran':
        tokenizer = Komoran()
    elif tokenizer_name == 'Mecab':
        tokenizer = Mecab()
    elif tokenizer_name == 'Okt':
        tokenizer = Okt()
    else:
        tokenizer = Kkma()

    words = re.split('\s+', text)

    tokens = []

    for word in words:
        if word == '':
            continue
        
        nouns = tokenizer.nouns(word.lower())

        if len(nouns) == 0:
            print(f'<{tokenizer_name}> {word} --> {nouns} !!! Failed. Use {word} anyway !!!')
            nouns.append(word)
        
        tokens = tokens + nouns
    
    return tokens

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

def tsne_plot(id_to_word, embeddings):
    labels = []
    tokens = []

    for id in range(len(id_to_word)):
        tokens.append(embeddings[:,id])
        labels.append(id_to_word[id])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
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
        p = np.reshape(softmax(y), -1)

        #return np.argmax(p)
        return np.argsort(p)[::-1][:3]

################################################################################
# Variables                                                                    #
################################################################################

epochs = 50

################################################################################
# Main                                                                         #
################################################################################

def main():
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <input_filepath> <predict_filepath> [plot]')
        return

    input_filepath = None
    predict_filepath = None
    if len(sys.argv) >= 3:
        input_filepath = sys.argv[1]
        predict_filepath = sys.argv[2]

    plot = False
    if len(sys.argv) >= 4 and sys.argv[3] == 'plot':
        plot = True

    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # Available names: Hannanum, Kkma, Komoran, Mecab, Okt
    #     * Mecab does not work at the moment due to installation problem
    tokenizer_name = 'Kkma'

    np.set_printoptions(precision=6)
    np.random.seed(42)
    
    # 1. Load data
    with open(input_filepath, mode='r', encoding='utf-8') as f:
        text = f.read()

    if predict_filepath is not None:
        with open(predict_filepath, mode='r', encoding='utf-8') as f:
            sample_words = f.readlines()

    # 2. Tokenize
    tokens = tokenize(tokenizer_name, text)
    word_to_id, id_to_word = mapping(tokens)
    
    window = 2
    X, y = generate_training_data(tokens, word_to_id, window)

    # 3. Split into train and test datasets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train, X_test, y_test = X, y, X, y

    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    #print(f'X_train[0]: {X_train[0]}')
    #print(f'np.argmax(X_train[0]): {np.argmax(X_train[0])} ({id_to_word[np.argmax(X_train[0])]})')
    #for i in range(X.shape[0]):
    #    print(f'[{i:4d}] {id_to_word[np.argmax(X[i,])]} {id_to_word[np.argmax(y[i,])]}')
    
    # 4. Fit with the train dataset
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

    # 5. Evaluate with the test dataset
    print('==== Test ====')
    loss, acc = mlp.train(X_test, y_test, False)
    print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

    # 6. Predict with the new input data
    print(f'===== Predict')
    for row in range(len(sample_words)):
        input_word = sample_words[row].strip()
        if len(input_word) == 0:
            continue
        measured = np.array(one_hot_encode(word_to_id[input_word], len(word_to_id)))
        target = mlp.predict(measured.reshape(-1, 1))
        print(f'**{input_word}** =>', end='')
        for idx in target:
            print(f' **{id_to_word[idx]}**', end='')
        print()

    tsne_plot(id_to_word, mlp.Wxh)

if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)
