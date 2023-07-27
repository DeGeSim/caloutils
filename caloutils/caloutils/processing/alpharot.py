import torch
from torch_scatter import scatter_mean

from fgsim.config import conf
from fgsim.utils import check_tensor

from .objcol import scaler


def rotate_alpha(alphas, batchidx, fake=False, center=False):
    batch_size = int(batchidx[-1] + 1)
    alphapos = conf.loader.x_features.index("alpha")
    ascalers = scaler.transfs_x[alphapos].steps[::-1]

    assert ascalers[0][0] == "standardscaler"
    mean = ascalers[0][1].mean_[0]
    scale = ascalers[0][1].scale_[0]

    # Backwards transform #0 stdscalar
    alphas = alphas.clone().double() * scale + mean

    # Backwards transform #1 logit
    alphas = torch.special.expit(alphas)

    assert (alphas <= 1).all()
    assert (0 <= alphas).all()
    # Rotation
    # smin, smax = ascalers[2][1].feature_range
    if not center:
        shift = torch.rand(batch_size).to(alphas.device)[batchidx]
    else:
        shift = -scatter_mean(alphas, batchidx)[batchidx] + 0.5
    if fake:
        shift *= 0
    alphas = alphas.clone() + shift
    alphas[alphas > 1] -= 1
    alphas[alphas < 0] += 1

    assert (alphas <= 1).all()
    assert (0 <= alphas).all()

    # Forward transform #1 logit
    alphas = torch.special.logit(alphas.clone())
    check_tensor(alphas)

    # Forward transform #0 stdscalar
    alphas = (alphas.clone() - mean) / scale
    check_tensor(alphas)
    return alphas.float()
