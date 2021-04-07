################################################################################
# Text sentiment classification                                                #
################################################################################

import os, sys, traceback
import numpy as np

################################################################################
# Functions                                                                    #
################################################################################

def make_inputs(text):
    inputs = []

    for w in text.split(' '):
        input = np.zeros((vocab_size, 1), dtype=int)
        if w in word_to_idx:
            input[word_to_idx[w]] = 1
        inputs.append(input)

    return inputs

def softmax(y):
    return np.exp((y)) / np.sum(np.exp(y))

################################################################################
# Classes                                                                      #
################################################################################

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size) / 1000
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_hs = {}

        h = np.zeros((self.Whh.shape[0], 1))
        self.last_hs[-1] = h

        for t, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[t] = h

        y = self.Why @ h + self.by

        return y, h

    def backword(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        d_Why = d_y @ self.last_hs[n-1].T
        d_by = d_y

        d_Wxh = np.zeros(self.Wxh.shape)
        d_Whh = np.zeros(self.Whh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T @ d_y

        for t in reversed(range(n)):
            temp = (1 - self.last_hs[t] ** 2) * d_h

            d_Wxh += temp @ self.last_inputs[t].T
            d_Whh += temp @ self.last_hs[t].T
            d_bh += temp

            d_h = self.Whh @ temp
            
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

    def train(self, data, backprop=True):
        loss = 0
        num_correct = 0

        for text, sentiment in data.items():
            inputs = make_inputs(text.lower())
            target = int(sentiment)
            y, h = self.forward(inputs)

            p = softmax(y)
            loss += -np.log(p[target])
            num_correct += int(np.argmax(p) == target)

            if backprop:
                d_y = p
                d_y[target] -= 1
                
                self.backword(d_y)

        loss = loss[0] / len(data)
        acc = num_correct / len(data)

        return loss, acc

    def predict(self, text):
        inputs = make_inputs(text.lower())
        y, h = self.forward(inputs)
        p = softmax(y)

        return np.argmax(p) == 1

################################################################################
# Variables                                                                    #
################################################################################

train_data = {
    "Great": True,
    "Bad": False,
    "Not bad": True,
    "Not good": False,
    "Exciting": True,
    "Not exciting": False,
    "Impressive": True,
    "Tedious": False,
    "CG looks good": True,
    "Its story is amazing": True,
    "Why do you waste your time": False,
    "It was hard to concentrate on some characters": False,
    "It is a good movie to watch with your children": True,
    "Action is not exciting": False,
    "Action is exciting": True,
    "The space scene is attractive": True,
    "The story was interesting": True,
    "Their jobs were interesting for me": True,
    "Their backgrounds are not clear": False,
    "I like the robot": True
}

test_data = {
    "CG looks great": True,
    "Not clear voice": False,
    "The role of the robot is very nice": True
}

sample_data = [
    "CG looks great",
    "Hard to understand their conversation",
    "Story is not exciting",
    "Story is exciting",
    "Nice robot"
]

vocab_size = 0
word_to_idx = {}
idx_to_word = {}

hidden_size = 128
output_size = 2
epochs = 2000

################################################################################
# Main                                                                         #
################################################################################

if __name__ == '__main__':
    try:
        vocab = list(set([w.lower() for text in train_data.keys() for w in text.split(' ')]))
        vocab_size = len(vocab)

        rnn = RNN(vocab_size, output_size, hidden_size)

        for i, w in enumerate(vocab):
            word_to_idx[w] = i
            idx_to_word[i] = w
        
        print('==== Train ====')
        for epoch in range(epochs):
            loss, acc = rnn.train(train_data, True)
            if epoch % 100 == 99:
                print(f'** Epoch: {epoch + 1} **')
                print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

        print('==== Test ====')
        loss, acc = rnn.train(test_data, True)
        print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')

        print('==== Predict ====')
        for text in sample_data:
            result = rnn.predict(text)
            print(f'{text} ==> {result}')
    except:
        traceback.print_exc(file=sys.stdout)
