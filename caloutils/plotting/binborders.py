import numpy as np
from fgsim.utils.torchtonp import wrap_torch_to_np


@wrap_torch_to_np
def binborders_wo_outliers(points: np.ndarray, bins=50) -> np.ndarray:
    assert len(points.shape) == 1
    if len(np.unique(points[:500])) < 50:
        uniques = np.sort(np.unique(points))
        if len(uniques) < 50:
            delta = (uniques[1] - uniques[0]) / 2
            return np.concatenate([uniques[0:1] - delta, uniques + delta])

    return np.linspace(*bounds_wo_outliers(points), num=bins, endpoint=True)


def bincenters(bins: np.ndarray) -> np.ndarray:
    return (bins[1:] + bins[:-1]) / 2


@wrap_torch_to_np
def bounds_wo_outliers(points: np.ndarray) -> tuple:
    median = np.median(points, axis=0)

    # med_abs_lfluk = np.sqrt(np.mean((points[points < median] - median) ** 2))
    # med_abs_ufluk = np.sqrt(np.mean((points[points > median] - median) ** 2))
    # upper = median + max(med_abs_ufluk,med_abs_ufluk)
    # lower = median - max(med_abs_ufluk,med_abs_ufluk)
    outlier_scale = (
        max(
            np.abs(np.quantile(points, 0.99) - median),
            np.abs(np.quantile(points, 0.01) - median),
        )
        * 1.1
    )
    upper = median + outlier_scale
    lower = median - outlier_scale
    # print(lower,np.min(points), upper,np.max(points))
    upper = np.min([upper, np.max(points)])
    lower = np.max([lower, np.min(points)])
    return lower, upper


def chip_to_binborders(arr, binborders):
    return np.clip(arr, binborders[0], binborders[-1])
