def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0])
    
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    out = [] 

    for i in range(out_h):
        current_row = []
        for j in range(out_w):
            
            max_val = -float('inf')
            
            for a in range(pool_size):
                for b in range(pool_size):
                    
                    h_idx = i * stride + a
                    w_idx = j * stride + b
                    
                    val = X[h_idx][w_idx]
                    
                    if val > max_val:
                        max_val = val
            
            current_row.append(max_val)
        
        out.append(current_row)

    return out