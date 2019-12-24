from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import time

benning = time.time()

digits_ = datasets.load_digits()

X = digits_.data
Y = digits_.target

np.random.seed(0)

n_samples = len(X)
percentage = 0.75

order = np.random.permutation(n_samples)
X = X[order]
Y = Y[order]

Y_test = Y[int(percentage * n_samples):]
X_test = X[int(percentage * n_samples):]

Y_treino = Y[:int(percentage * n_samples)]
X_treino = X[:int(percentage * n_samples)]

clf = GaussianNB(var_smoothing=9e-1)
clf.fit(X_treino, Y_treino)
predict_ = clf.predict(X_test)
print('\nBase:')
print(predict_)

accuracy = clf.score(X_test, Y_test)
print('\nPrecisão: ', accuracy*100, '%')

array = metrics.confusion_matrix(Y_test, predict_)
print('\nArray:')
for item in array:
    print(item)

