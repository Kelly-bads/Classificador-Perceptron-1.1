from sklearn import svm, neural_network, metrics, datesets
import numpy as np 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


db = datesets.load_boston()

x = db.data 
y = db.target

np.random.seed(0)

amostras = len(x)
divisao = 0.75

embaralhar = np.random.permutation(amostras)

x = x[embaralhar]
y= y[embaralhar]

x_treino = x [:int(amostras*divisao)]
y_treino = y [:int(amostras*divisao)]

x_teste = x [int(amostras*divisao):]
y_teste = y [int(amostras*divisao):]

parametros_svr = {'kernel':('linear','poly','sigmoid','rbf'),'C':[1,2,3,4,5]}

svr = svm.SVR()

clf = GridSearchCV(svr,parametros_svr, n_jobs=10)

print(clf.best_params_)

clf = svm.SVR(kernel = 'linear')

clf.fir(x_treino, y_treino)

predicao = clf.predict(x_teste)

mse = metrics.mean_squared_error(y_teste, predicao)

r2 = metrics.r2_score(y_teste, predicao)

print('MSE:', mse)
print('R2:', r2)

diagonal = list(range(int(min(y_teste)), int(max(y_teste))))


plt.scatter(y_teste,predicao)
plt.plot(diagonal,diagonal,'r--')
plt.show()