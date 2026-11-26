"""Training-time logit adjustment used in the R52 study."""
import torch
from torch.nn import functional as F


def log_train_prior(train_labels: torch.Tensor, classes: int) -> torch.Tensor:
    """Return log class probabilities estimated from training labels."""
    counts = torch.bincount(train_labels, minlength=classes)
    if counts.numel() != classes or (counts == 0).any():
        raise ValueError("Every class must have training examples.")
    return (counts.double() / counts.sum()).log().float()


def adjusted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_prior: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Return summed cross-entropy on adjusted training logits."""
    return F.cross_entropy(logits + tau * log_prior.to(logits.device), labels, reduction="sum")
