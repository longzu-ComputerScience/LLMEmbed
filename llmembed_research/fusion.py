"""Parameter-free routing change evaluated in the hierarchical fusion pilot."""
from collections.abc import Callable

import torch


def hierarchical_gate_weights(
    sources: torch.Tensor,
    score: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return seven effective weights for five LLaMA layers and two encoders."""
    if sources.ndim != 3 or sources.shape[1] != 7:
        raise ValueError("Expected [batch, 7 sources, hidden] input.")
    layers, encoders = sources[:, :5], sources[:, 5:]
    layer_weights = score(layers).softmax(dim=1)
    llama = (layer_weights * layers).sum(dim=1, keepdim=True)
    backbone_weights = score(torch.cat((llama, encoders), dim=1)).softmax(dim=1)
    return torch.cat((layer_weights * backbone_weights[:, :1], backbone_weights[:, 1:]), dim=1)
