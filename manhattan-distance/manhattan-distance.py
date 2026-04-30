import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x_array = np.array(x)
    y_array = np.array(y)
    
    absolute_differences = np.abs(x_array - y_array)
    
    result = np.sum(absolute_differences)
    
    return float(result)