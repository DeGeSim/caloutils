import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_std


def sphereratio(batch: Batch) -> dict[str, torch.Tensor]:
    """
    Computes energy fractions within a sphere around the center of energy for each event in a batch,
    and returns the ratios of these energy fractions.

    The function calculates the energy fractions within two spheres (small and large) around the center
    of energy for each event in a batch. The centers are calculated as a weighted mean of the point
    coordinates, where the weights are the energies of the points.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events.

    Returns
    -------
    dict
        A dictionary with keys 'small', 'large', and 'ratio', where:
            - 'small' corresponds to the tensor of energy fractions within the smaller sphere for each event.
            - 'large' corresponds to the tensor of energy fractions within the larger sphere for each event.
            - 'ratio' is the ratio of 'small' to 'large' for each event.
    """
    batchidx = batch.batch
    # Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)
    Ehit = batch.x[:, 0].reshape(-1, 1)
    e_small, e_large = __dist_fraction(Ehit, batch.xyz, batchidx, 0.3, 0.8)

    return {
        "small": e_small,
        "large": e_large,
        "ratio": e_small / e_large,
    }


def cyratio(batch: Batch) -> dict[str, torch.Tensor]:
    """
    Similar to the sphereratio function, this function computes energy fractions within a cylinder around
    the center of energy for each event in a batch, and returns the ratios of these energy fractions.

    The function only considers the x and y coordinates of the points to calculate the energy fractions
    and the center of energy, effectively projecting the points onto the xy-plane and creating a cylindrical
    region.

    Parameters and return values are the same as those for the sphereratio function.
    """
    batchidx = batch.batch
    # Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)
    Ehit = batch.x[:, 0].reshape(-1, 1)

    e_small, e_large = __dist_fraction(
        Ehit, batch.xyz[:, [0, 1]], batchidx, 0.2, 0.6
    )

    return {
        "small": e_small,
        "large": e_large,
        "ratio": e_small / e_large,
    }


def __dist_fraction(Ehit, pos, batchidx, small, large, center_energy_weighted=True):
    """
    Private helper function used by the sphereratio and cyratio functions to compute the energy fractions
    within certain regions around the center of energy.

    Parameters
    ----------
    Ehit : torch.Tensor
        Tensor of energies for each point in the batch.
    pos : torch.Tensor
        Tensor of positions for each point in the batch.
    batchidx : torch.Tensor
        Tensor of batch indices for each point in the batch.
    small : float
        Radius of the smaller region.
    large : float
        Radius of the larger region.
    center_energy_weighted : bool, optional
        Whether to calculate the center of energy as a weighted mean of the point coordinates.
        If False, the unweighted mean is used. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing two tensors:
            - The energy fraction within the smaller region for each event.
            - The energy fraction within the larger region for each event.
    """
    Esum = global_add_pool(Ehit, batchidx).reshape(-1, 1)

    # get the center, weighted by energy
    if center_energy_weighted:
        center = global_add_pool(pos * Ehit, batchidx) / Esum
    else:
        center = global_mean_pool(pos, batchidx)
    std = scatter_std(pos, batchidx, dim=-2)
    # hit distance to center
    delta = (((pos - center[batchidx]) / std[batchidx]) ** 2).mean(-1).sqrt()
    del center, std
    # energy fraction inside circle around center
    e_small = (
        global_add_pool(Ehit.squeeze() * (delta < small).float(), batchidx)
        / Esum.squeeze()
    )
    e_large = (
        global_add_pool(Ehit.squeeze() * (delta < large).float(), batchidx)
        / Esum.squeeze()
    )
    # __plot_frac(e_small, e_large, small, large)
    return e_small, e_large
