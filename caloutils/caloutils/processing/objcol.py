from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from torch_geometric.data import Data




def path_to_len(fn: Path) -> int:
    return len(h5py.File(fn, "r")["incident_energies"])




def readpath(
    fn: Path,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(fn, "r") as electron_file:
        energies = electron_file["incident_energies"][start:end]
        showers = electron_file["showers"][start:end]
    res = (torch.Tensor(energies), torch.Tensor(showers))
    return res


def read_chunks(
    chunks: List[Tuple[Path, int, int]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    res = (
        torch.concat([e[0] for e in chunks_list]),
        torch.concat([e[1] for e in chunks_list]),
    )
    return [(res[0][ievent], res[1][ievent]) for ievent in range(len(res[1]))]


def contruct_graph_from_row(chk: Tuple[torch.Tensor, torch.Tensor]) -> Data:
    E, shower = chk[0].clone(), chk[1].clone()
    num_z = 45
    num_alpha = 16
    num_r = 9
    shower = shower.reshape(num_z, num_alpha, num_r)
    idxs = torch.where(shower)
    h_energy = shower[idxs]
    z, alpha, r = idxs

    pc = torch.stack([h_energy, z, alpha, r]).T
    assert not pc.isnan().any()
    num_hits = torch.tensor(idxs[0].shape).float()
    res = Data(x=pc, y=torch.concat([E, num_hits, E / num_hits]).reshape(1, 3))
    return res


def Identity(x):
    return x


def LimitForBoxCox(x):
    return np.clip(x, -19, None)


# hitE_tf = make_pipeline(
#     PowerTransformer(method="box-cox", standardize=False),
#     FunctionTransformer(Identity, LimitForBoxCox, validate=True),
#     # SplineTransformer(),
#     StandardScaler(),
# )

# E_tf = make_pipeline(
#     PowerTransformer(method="box-cox", standardize=False),
#     QuantileTransformer(output_distribution="normal"),
# )
# num_particles_tf = make_pipeline(
#     *dequant_stdscale((0, conf.loader.n_points + 1)),
#     QuantileTransformer(output_distribution="normal"),
# )
# E_per_hit_tf = make_pipeline(
#     PowerTransformer(method="box-cox", standardize=False),
#     QuantileTransformer(output_distribution="normal"),
# )

# scaler = ScalerBase(
#     files=file_manager.files,
#     len_dict=file_manager.file_len_dict,
#     transfs_x=[
#         hitE_tf,  # h_energy
#         make_pipeline(*dequant_stdscale()),  # z
#         make_pipeline(*dequant_stdscale()),  # alpha
#         make_pipeline(*dequant_stdscale()),  # r
#     ],
#     transfs_y=[
#         E_tf,  # Energy
#         num_particles_tf,  # num_particles
#         E_per_hit_tf,  # Energy pre hit
#     ],
#     read_chunk=read_chunks,
#     transform_wo_scaling=contruct_graph_from_row,
# )
