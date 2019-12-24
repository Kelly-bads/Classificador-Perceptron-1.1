from sklearn import metrics, datasets
from sklearn.linear_model import Perceptron

X, y = datasets.load_digits(return_X_y=True)
clf = Perceptron(alpha=0.1, max_iter=50000, penalty=None)
clf.fit(X, y)

predict_ = clf.predict(X)
print('\nBase:')
print(predict_)

array = metrics.confusion_matrix(y, predict_)
print('\nArray:')
for item in array:
    print(item)

precisao = clf.score(X, y)
print('\nPrecisão: ', precisao * 100, '%')


