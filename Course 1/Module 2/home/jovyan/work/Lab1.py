import numpy as np    # it is an unofficial standard to use np for numpy
import time

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.)
print(
    f"\nnp.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(
    f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# NumPy routines which allocate memory and fill with user specified values
a = np.array([5, 4, 3, 2])
print(
    f"\nnp.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5., 4, 3, 2])
print(
    f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")


# vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# access an element
print(
    f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

# indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
