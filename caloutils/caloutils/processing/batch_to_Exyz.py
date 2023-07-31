from torch_geometric.data import Batch
def batch_to_Exyz(batch: Batch) -> Batch:
    """
    Transforms the cylindrical coordinate representation (z, alpha, r) in a batch of graphs to Cartesian coordinates
    (x, y, z), and calculates pseudo-rapidity (eta) and azimuthal angle (phi) for each node in the graph.

    Parameters
    ----------
    batch : Batch
        The batch of graphs where each node feature vector includes a cylindrical coordinate representation
        (Energy, z, alpha, r) where z is the long axis of the cylindrical coordinate system, alpha is the
        azimuthal angle, and r is the radial distance from the z-axis.

    Returns
    -------
    batch : Batch
        The modified batch object where an additional "xyz" attribute is added which includes the Cartesian coordinates
        for each node in the graphs. Additionally, "eta" (pseudo-rapidity) and "phi" (azimuthal angle) attributes are
        calculated and added to the batch object.

    Note
    ----
    Pseudo-rapidity (eta) and azimuthal angle (phi) are commonly used in particle physics to describe the direction
    of a particle relative to the beam axis (z-axis in this context).
    """
    zalphar: torch.Tensor = batch.x[..., [1, 2, 3]]
    num_alpha=calorimeter.num_alpha
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
