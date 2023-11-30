import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-darkgrid')

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)
fig, ax = plt.subplots(figsize=(20,7))
plt.scatter(x,y)

x_b = np.c_[np.ones((100,1)),x]

theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
y_hat = theta[0] + x * theta[1]

fig, ax = plt.subplots(figsize=(20,7))
plt.plot(x, y_hat)
plt.scatter(x,y)


eta = 0.1
n_interations = 10 
m = 100

theta = np.random.randn(2,1)
fig, ax = plt.subplots(figsize=(25,8))
plt.scatter(x,y)

for interation in range(n_interations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta)-y)
    theta = theta - eta * gradients
    y_hat = theta[0] + x*theta[1]

    ax.plot(x, y_hat, alpha=0.3 + interation / n_interations, color='blue')


#REGRESSAO POLINOMINAL
m = 100
x = 6*np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 * np.random.randn(m, 1)

plt.scatter(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

x_poly = poly.fit_transform(x)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

lin_reg.intercept_

lin_reg.coef_

plt.scatter(x,y)
plt.scatter(x, lin_reg.predict(x_poly))

for degree in [1,2,10]:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    x_seq = np.linspace(x.min(), x.max(), 300).reshape(-1,1)

    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)

    plt.scatter(x,y)
    x_seq_transf = poly_features.fit_transform(x_seq)
    plt.plot(x_seq, lin_reg.predict(x_seq_transf), label= degree)
ax.set_ylim([-5, 12])
ax.set_xlim([-4, 4])

#dados aleatorios
m =100
x = np.random.rand(m, 1) - 3
y = np.random.randn(m, 1)
plt.scatter(x,y)

from sklearn.linear_model import Ridge


fig, ax = plt.subplots(figsize=(25,8))
plt.scatter(x, y)
for alpha in [ 0, 1e-5, 1]:
    poly_features = PolynomialFeatures(degree=10,include_bias=False)
    x_poly = poly_features.fit_transform(x)
    x_seq = np.linspace(x.min(), x.max(),300).reshape(-1, 1)

    ridge_reg = Ridge(alpha=alpha, solver='cholesky')
    ridge_reg.fit(x_poly, y)

    x_seq_transf = poly_features.fit_transform(x_seq)
    plt.plot(x_seq, ridge_reg.predict(x_seq_transf), label=alpha)

ax.legend()


#REGRESSAO LOGISTICA

from sklearn import datasets

iris = datasets.load_iris()
iris['data'].shape

iris['target_names']

x = iris['data'][:, 3:]#pegando todas as linhas mas so a 3 column
y = (iris['target'] == 2).astype(np.int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x, y)

fig, ax = plt.subplots(figsize=(25, 8))

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = log_reg.predict_proba(x_new)

plt.plot(x_new, y_prob[:, 1])

x = iris['data'][:, (2,3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial')
softmax_reg.fit(x, y)

softmax_reg.predict([[5,2]])

softmax_reg.predict_proba([[5,2]])