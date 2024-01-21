import numpy as np

# Load our data set
# x_train = np.array([1.0, 2.0])  # features
# y_train = np.array([300.0, 500.0])  # target value

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([5, 7, 9, 11, 13])


# Function to calculate the cost


# def compute_cost(x, y):

#     m = x.shape[0]
#     cost = 0

#     for i in range(m):
#         f_wb = w * x[i] + b
#         cost = cost + (f_wb - y[i])**2
#     total_cost = 1 / (2 * m) * cost

#     return total_cost


def compute_gradient(x, y):

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    iteration = 1000
    alpha = 0.1

    for i in range(iteration):
        f_wb = dj_dw * x + dj_db

        # ! Calculates the cost function
        cost = (1/m) * sum(vai ** 2 for vai in (f_wb - y))

        dj_dw_i = 1/m * (sum(x * (f_wb - y)))
        dj_db_i = 1/m * (sum(f_wb - y))

        dj_db = dj_db - alpha * dj_db_i  # ! b = b - learning rate * d/db
        dj_dw = dj_dw - alpha * dj_dw_i  # ! w = w - learning rate * d/dw

        print("w {} , b {} , cost {} , iteration {}".format(dj_dw, dj_db, cost, i))
 

compute_gradient(x_train, y_train)
