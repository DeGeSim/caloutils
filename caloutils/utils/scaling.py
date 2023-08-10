import torch


def scale_b_to_a(a: torch.Tensor, b: torch.Tensor):
    """
    Scale tensor b to the same mean and standard deviation as tensor a.

    Args:
        a (torch.Tensor): Target tensor for scaling.
        b (torch.Tensor): Source tensor to be scaled.

    Returns:
        torch.Tensor: Scaled version of tensor b.
    """
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb
