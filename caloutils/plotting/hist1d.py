from typing import Dict

from fgsim.config import conf
from fgsim.plot.infolut import var_to_label
from matplotlib.figure import Figure

from .ratioplot import ratioplot


def hist1d(
    sim, gen, ftxname: str, bins=None, energy_weighted=False
) -> Dict[str, Figure]:
    epos = conf.loader.x_ftx_energy_pos
    ename = conf.loader.x_features[epos]

    sim_features = _exftxt(sim[ftxname])
    gen_features = _exftxt(gen[ftxname])
    fext = "_Ew" if energy_weighted else ""

    plots_d: Dict[str, Figure] = {}
    for iftx, ftn in enumerate(conf.loader.x_features):
        if energy_weighted and iftx == epos:
            continue
        b = bins[iftx] if bins is not None else None
        simw = sim_features[ename] if energy_weighted else None
        genw = gen_features[ename] if energy_weighted else None
        plots_d[f"marginal_{ftn}{fext}.pdf"] = ratioplot(
            sim=sim_features[ftn],
            gen=gen_features[ftn],
            title=var_to_label(ftn),
            bins=b,
            simw=simw,
            genw=genw,
        )

    return plots_d


def _exftxt(arr):
    return {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features,
            arr.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
