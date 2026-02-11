import torch
import torch.nn.functional as F
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    tensor_scores = torch.tensor(scores, dtype=torch.float32)

    return F.log_softmax(tensor_scores, dim=0)
