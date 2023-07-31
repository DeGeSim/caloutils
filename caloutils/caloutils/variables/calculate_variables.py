from torch_geometric.data import Batch
from .analyze_layers import analyze_layers
from .energy_ratios import sphereratio, cyratio, __dist_fraction
from .response import response
from .first_principal_components import first_principal_components
from ..processing.utils import sum_duplicate_hits
def calc_vars(batch:Batch,fake: bool=False,vars:list=["voxel","sphereratio","cyratio","fpc","showershape","response"]):
    """
    Calculates specified variables for each event in a batch.

    The function supports calculation of "sphereratio", "cyratio", "fpc", "showershape",
    and "response". Calculated variables are added to the input batch object.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events.
    fake : bool, optional
        If True, sums up the energy of duplicate hits in an event. Defaults to False.
    vars : list of str, optional
        List of variables to calculate for the events in the batch. Defaults to all
        supported variables: ["voxel","sphereratio","cyratio","fpc","showershape","response"].

    Returns
    -------
    Batch
        The input Batch object with calculated variables added.
    """
    if fake:
        batch = sum_duplicate_hits(batch)
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