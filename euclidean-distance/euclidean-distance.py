import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x_array =  np.asarray(x , dtype = float)
    y_array = np.asarray(y , dtype = float)
    result = np.sqrt(np.sum(np.square(x_array-y_array)))
    return result