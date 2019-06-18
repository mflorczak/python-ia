import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self, n):
        self.w = np.random.randint(2, size=n + 1) * 1.0
        self.w[self.w < 1.0] = -1.0
        self.w[0] = 1.0

    def predict(self, x):
        return int(np.dot(x, self.w[1:]) + self.w[0] >= 0)

    def train(self, xx, d, eta, tol):
        while self.evaluate_test(xx, d)[0] < tol:
            for inputs, label in zip(xx, d):
                prediction = self.predict(inputs)
                if prediction != label:
                    self.w[1:] += eta * (label - prediction) * inputs
                    self.w[0] += eta * (label - prediction)

    def evaluate_test(self, xx, d):
        t = 0
        predictions = []
        for input, correct in zip(xx, d):
            predictions.append(self.predict(input))
            if self.predict(input) == correct:
                t += 1
        return t / len(d), predictions


data = pd.read_csv('2D.csv', sep=';', header=None, usecols=[0, 1, 2])

xx = []
d = []

for firstCol, secCol, threeCol in zip(data[0][1:], data[1][1:], data[2][1:]):
    xx.append([firstCol.replace(',', '.'), secCol.replace(',', '.')])
    d.append(threeCol)

xx = np.array(xx).astype(np.float)

perceptron = Perceptron(2)
perceptron.train(xx[:400], d[:400], 0.01, 1)

for x, d in zip(xx[400:], d[400:]):
    incorrect = 0
    if perceptron.predict(x) != d:
        incorrect += 1
    print(perceptron.predict(x), d)
print('Incorrect learning: ', incorrect)
