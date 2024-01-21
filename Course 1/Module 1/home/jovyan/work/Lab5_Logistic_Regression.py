import numpy as np

def sigmoid(x):
    
    x = 1 / (1 + np.exp(-x))

    return x

def logistic_cost(x, y, w, b):

    m = x.shape[0]
    sum_cost = 0

    for i in range(m):
        z = np.dot(w, x[i]) + b
        y_pred = sigmoid(z)
        
        if(y[i] == 1):
            sum_cost += np.log(y_pred)
        else:
            sum_cost += np.log(1 - y_pred)

    cost = sum_cost / (-m)
    
    return cost

def compute_gradient_logistic(x, y, w, b):
    
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        z = np.dot(w, x[i]) + b
        y_pred = sigmoid(z)
        err = y_pred - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i][j]
        dj_db += err 
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, iters):

    m = x.shape[0]
    w = w_in
    b = b_in

    for i in range(iters):
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if(i % 1000 == 0):
            print(f"Iteration {i:4d}: Cost {logistic_cost(x, y, w, b)}")

    return w, b

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")