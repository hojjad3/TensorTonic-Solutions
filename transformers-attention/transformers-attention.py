import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    attention_score  = Q @ torch.transpose(K, -2, -1)
    scaled = attention_score  /   math.sqrt(K.size(-1))
    A = F.softmax(scaled,dim  = -1)
    output = A @ V 
    return output 