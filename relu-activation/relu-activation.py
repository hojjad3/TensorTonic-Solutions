import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x_array = np.array(x)
    relu_output = np.maximum(0, x_array)
    return relu_output