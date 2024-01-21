import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot() * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_prediction(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(pridictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iter, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iter):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 50 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_prediction(A2), Y))

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# def ReLU_deriv(Z):
#     return Z >= 0


# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     one_hot_Y = one_hot_Y.T
#     return one_hot_Y


# def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
#     one_hot_Y = one_hot(Y)
#     m = Y.shape[0]
#     loss = -np.sum(one_hot_Y * np.log(A2)) / m
#     dZ2 = A2 - one_hot_Y
#     dW2 = 1 / m * dZ2.dot(A1.T)
#     db2 = 1 / m * np.sum(dZ2)
#     dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
#     dW1 = 1 / m * dZ1.dot(X.T)
#     db1 = 1 / m * np.sum(dZ1)
#     return dW1, db1, dW2, db2, loss


# def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
#     W1 = W1 - alpha * dW1
#     b1 = b1 - alpha * db1
#     W2 = W2 - alpha * dW2
#     b2 = b2 - alpha * db2
#     return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


# def get_accuracy(predictions, Y):
#     print(predictions, Y)
#     return np.sum(predictions == Y) / Y.size


# def gradient_descent(X, Y, alpha, iterations):
#     W1, b1, W2, b2 = init_params()
#     for i in range(iterations):
#         Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
#         dW1, db1, dW2, db2, loss = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
#         W1, b1, W2, b2 = update_params(
#             W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if i % 50 == 0:
#             print("Iteration:", i)
#             predictions = get_predictions(A2)
#             accuracy = get_accuracy(predictions, Y)
#             print("Accuracy:", accuracy)
#             print("Loss:", loss)
#     return W1, b1, W2, b2


# W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.05, 1000)

# np.save('W1.npy', W1)
# np.save('b1.npy', b1)
# np.save('W2.npy', W2)
# np.save('b2.npy', b2)

# params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
# np.save('model_parameters.npy', params)


loaded_W1 = np.load('W1.npy')
loaded_b1 = np.load('b1.npy')
loaded_W2 = np.load('W2.npy')
loaded_b2 = np.load('b2.npy')

loaded_params = np.load('model_parameters.npy', allow_pickle=True).item()
loaded_W1 = loaded_params['W1']
loaded_b1 = loaded_params['b1']
loaded_W2 = loaded_params['W2']
loaded_b2 = loaded_params['b2']


img_num = 1

while os.path.isfile(f"digits/digit{img_num}.png"):
    try:
        img = cv2.imread(f"digits/digit{img_num}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        Z1, A1, Z2, A2 = forward_prop(
            loaded_W1, loaded_b1, loaded_W2, loaded_b2, img.flatten())
        predicted_digit = get_predictions(A2)[0]

        print(f"This image is probably a {predicted_digit}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.show()
    except Exception as e:
        print(f'Error: {e}')
    finally:
        img_num += 1

print(loaded_W1, loaded_W2, loaded_b1, loaded_b2)
