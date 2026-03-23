import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    tp_count = np.sum(y_true_arr == y_pred_arr)
    return float(tp_count / len(y_true))
    
    