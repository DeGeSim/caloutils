from torch_geometric.data import Batch

from ..processing import shift_multi_hits, shift_sum_multi_hits, sum_multi_hits
from ..processing.batch_to_Exyz import batch_to_Exyz
from .analyze_layers import analyze_layers
from .energy_ratios import cyratio, sphereratio
from .first_principal_components import first_principal_components
from .response import response


def calc_vars(
    batch: Batch,
    sum_multihits: bool = True,
    shift_multihits: bool = False,
    vars: list = [
        "voxel",
        "sphereratio",
        "cyratio",
        "fpc",
        "showershape",
        "response",
    ],
):
    """
    Calculates specified variables for each event in a batch.

    The function supports calculation of "sphereratio", "cyratio", "fpc", "showershape",
    and "response". Calculated variables are added to the input batch object.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events.
    sum_multihits : bool, optional
        If True, aggregates the energies of hits in the same cell for each event. Defaults to True.
    shift_multihits : bool, optional
        If True, tries to shift hits that occupy the same cell to neihgboring, empty cells. Defaults to False.
    vars : list of str, optional
        List of variables to calculate for the events in the batch. Defaults to all
        supported variables: ["voxel","sphereratio","cyratio","fpc","showershape","response"].

    Returns
    -------
    Batch
        The input Batch object with calculated variables added.
    """
    if sum_multihits and shift_multihits:
        batch = shift_sum_multi_hits(batch)
    elif sum_multihits:
        batch = sum_multi_hits(batch)
    elif shift_multihits:
        batch = shift_multi_hits(batch)

    batch = batch_to_Exyz(batch)
    if "sphereratio" in vars:
        batch["sphereratio"] = sphereratio(batch)
    if "cyratio" in vars:
        batch["cyratio"] = cyratio(batch)
    if "fpc" in vars:
        batch["fpc"] = first_principal_components(batch)
    if "showershape" in vars:
        batch["showershape"] = analyze_layers(batch)
    if "response" in vars:
        batch["response"] = response(batch)
    return batch
