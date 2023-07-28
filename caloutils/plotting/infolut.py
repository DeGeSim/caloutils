from typing import Dict, Optional, Union

import numpy as np
from fgsim.config import conf

labels_dict: Dict[str, str] = {
    "etarel": "$\\eta_\\mathrm{rel}$",
    "phirel": "$\\phi_\\mathrm{rel}$",
    "ptrel": "$p^\\mathrm{T}_\\mathrm{rel}$",
    "mass": "$m_{rel}$",
    "phi": "$Σ ϕ_{rel}$",
    "pt": "$Σp_{T}^{rel}$",
    "eta": "$Ση_{rel}$",
    "response": "Response (E/ΣE)",
    "showershape_peak_layer": "Peak Layer",
    "showershape_psr": "Ratio Δ(Turnon(/off), Peak) Layer",
    "showershape_turnon_layer": "Turnon Layer",
    "sphereratio": "Sphere(Δ=0.2)/Sphere(Δ=0.3)",
    "fpc_x": "First PCA vector x",
    "fpc_y": "First PCA vector y",
    "fpc_z": "First PCA vector z",
    "fpc_eval": "First PCA Eigenvalue",
}


def var_to_label(v: Union[str, int]) -> str:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v
    if vname in labels_dict:
        return labels_dict[vname]
    else:
        return vname


def var_to_bins(v: Union[str, int]) -> Optional[np.ndarray]:
    if isinstance(v, int):
        vname = conf.loader.x_features[v]
    else:
        vname = v

    if (
        conf.dataset_name == "calochallange"
        and "calochallange2" in conf.loader.dataset_path
    ):
        return {
            "E": np.linspace(0, 6000, 100 + 1) - 0.5,  # E
            "z": np.linspace(0, 45, 45 + 1) - 0.5,  # z
            "alpha": np.linspace(0, 16, 16 + 1) - 0.5,  # alpha
            "r": np.linspace(0, 9, 9 + 1) - 0.5,  # r
        }[vname]
    return None
