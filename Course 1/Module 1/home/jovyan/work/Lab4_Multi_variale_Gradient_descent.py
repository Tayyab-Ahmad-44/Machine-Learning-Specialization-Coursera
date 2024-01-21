import numpy as np
import time
import math

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

m = 3 #! No of training examples

f_wb = np.zeros(m)

def compute_cost(X, Y, W, B):
    M = X.shape[0]
    summation = 0
    for i in range(0,M):
        summation += ((np.dot(W, X[i]) + B) - Y[i]) ** 2
    
    Cost = summation / (2 * M)
    
    return Cost

def compute_gradient(X, Y, W, B):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(W, X[i]) + B) - Y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def grad_desc(X, Y, W_init, B_init, cost_func, grad_func, alpha, iters):
    
    J_history = []
    w = W_init
    b = B_init

    for i in range(iters):
        
        dj_dw, dj_db = grad_func(X, Y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(cost_func(X, Y, w, b))

        if i% math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history


init_w = np.zeros_like(w_init)
init_b = 0.
alpha = 5.0e-7
iteration = 1000
w_final, b_final, J = grad_desc(X_train, y_train, init_w, init_b, compute_cost, compute_gradient, alpha, iteration)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")