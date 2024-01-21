import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

x_train = [1, 2]  # ! Size in 1000 sqft
y_train = [300, 500]  # ! price in 1000s of dollar


def compute_cost(x, y, w, b):
    m = x.shape[0]  # ! Determine the size of the array received
    cost_sum = 0

    for i in range(m):
        # ! total cost = 1/2m ( summation_from_0_till_m( (f_wb - y[i]) ** 2) )
        f_wb = (w * x[i]) + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


plt_intuition(x_train, y_train)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730])

# ! It ensures that any old plots are closed before a new one is created
plt.close('all')
# ! It creates a new figure and axis
fig, ax, dyn_items = plt_stationary(x_train, y_train)
# ! It allows the user to interact with the plot and update it as needed.
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()
