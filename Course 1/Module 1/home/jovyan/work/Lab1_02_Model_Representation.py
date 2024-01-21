import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2])
y_train = np.array([300, 500])

w = 300
b = 0

def compute_model_output(x, w, b):
    m = x.shape[0] #! No of example in data set
    f_wb = np.zeros(m)  #! Empty array of size m
    for i in range(m):
        f_wb[i] = w * x[i] + b #! A linear Function
        
    return f_wb

temp_f_wb = compute_model_output(x_train, w, b)

#! Plot our model prediction
plt.plot(x_train, temp_f_wb, c='b', label='Our Prediction')

#! Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

#! Set the title
plt.title("Housing Prices")
#! Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
#! Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
