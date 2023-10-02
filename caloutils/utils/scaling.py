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
    assert a.dim() == b.dim() == 1
    mean, std = a.mean(), a.std()
    sa = a - mean
    sb = b - mean

    if torch.log10(std) < -8:
        std = 1
    scale_a = torch.log10(sa.abs().mean()) - torch.log10(std)
    scale_b = torch.log10(sb.abs().mean()) - torch.log10(std)
    if scale_a > 10 or scale_b > 10:
        std += 1e-6

    sa = sa / std
    sb = sb / std
    return sa, sb
