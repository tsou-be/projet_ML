import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import copy
import math
from utils import *
# Note: utils.py doit être dans le même répertoire ou chemin approprié

# Fonction sigmoid
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

# Fonction coût logistique
def compute_cost(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    total_cost = cost / m
    return total_cost

# Fonction gradient logistique
def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw

# Fonction descente de gradient
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            cost = cost_function(X, y, w, b, lambda_)
            J_history.append(cost)
        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w.copy())
            print(f'Iteration {i:4}: Cost {J_history[-1]:8.2f}')
    return w, b, J_history, w_history

# Fonction prédiction
def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0
    return p

# Chargement données (remplacez par vos chemins de fichiers)
X_train, y_train = load_data("data/ex2data1.txt")

# Exemple utilisation (première partie)
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.
iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

# Prédictions et précision
p = predict(X_train, w, b)
print(f'Train Accuracy: {np.mean(p == y_train) * 100:.2f}%')

# Plot (utilise fonction helper de utils.py)
plot_decision_boundary(w, b, X_train, y_train)

# Note: Pour la partie regularisée, implémentez compute_cost_reg et compute_gradient_reg de manière similaire
# en ajoutant les termes de régularisation comme indiqué dans le notebook.
# Les données ex2data2.txt manquent, chargez-les séparément.

