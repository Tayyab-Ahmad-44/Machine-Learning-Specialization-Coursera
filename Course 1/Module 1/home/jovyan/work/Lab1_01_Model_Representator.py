import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')

#! x_train is the input variable (Area in 1000 Square Feet)
#! y_train is the output variable (Pricce in 1000s of Dollar)

x_train = np.array([1, 2, 3])
y_train = np.array([300, 500, 800])

print(f"\nx_train = {x_train}")
print(f"y_train = {y_train}")


#! m is te number of training examples.

print(f"\nx_train.shape: {x_train.shape}")
m = len(x_train)
print(f"Number of training example: {m}")


#! $i^{th}$ training example.

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"\nx^({i}), y^({i}) = ({x_i}, {y_i})\n")


# #! Plot the data points
# plt.scatter(x_train, y_train, marker = 'x', c = 'r')
# #! Set the title
# plt.title("Housing Prices")
# #! Set the y-axis lable
# plt.ylabel('Price (in 1000s of dollars)')
# #! Set th x-axis label
# plt.ylabel('Size (1000 Sqft)')
# #! Display
# plt.show()
