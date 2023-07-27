import torch
from torch_geometric.data import Batch


def batch_to_Exyz(batch: Batch) -> Batch:
    zalphar: torch.Tensor = batch.x[..., [1, 2, 3]]
    num_alpha = 16
    # num_z = 45
    # num_r = 9
    z, alpha, r = zalphar.T
    # shift idx by one to go from (min,max)
    # (0, 1-1/num) -> (1/num,1)
    # z = (z + 1) / num_z
    # r = (r + 1) / num_r
    alpha = (alpha + 1) / num_alpha * torch.pi * 2

    y = r * torch.cos(alpha)
    x = r * torch.sin(alpha)

    batch.xyz = torch.stack([x, y, z]).T

    #       r
    #       ^
    #     / |
    #    /  |
    #   /   |
    # / θ)  |
    # ------> z

    theta = torch.arctan(r / z)
    if (theta.abs() > torch.pi).any():
        raise RuntimeError("θ not in forward direction")
    if (theta < 0).any():
        raise RuntimeError("θ not in forward direction")
    batch.eta = -torch.log(torch.tan(theta / 2.0)).reshape(-1, 1)
    batch.phi = alpha.reshape(-1, 1)

    return batch
