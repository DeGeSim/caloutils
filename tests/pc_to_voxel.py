# %%
import numpy as np
import torch
from fgsim.config import compute_conf, conf
from torch_geometric.data import Batch
from tqdm import tqdm

compute_conf(conf, {"dataset_name": "calochallange", "command": "test"})
step = 1_000
conf.loader.scaling_fit_size //= 10
conf.loader.batch_size = step

from fgsim.io.dequantscaler import (  # noqa: E402
    FunctionTransformer,
    dequant,
    requant,
    scipy,
)
from fgsim.loaders.calochallange.objcol import (  # noqa: E402
    contruct_graph_from_row,
    file_manager,
    read_chunks,
    scaler,
)
from fgsim.loaders.calochallange.pc_to_voxel import (  # noqa: E402
    dims,
    pc_to_voxel,
    sum_multi_hits,
)

# scaler.fit(True)
# print("fit done")

for iftx in range(1, 4):
    sel_tf = scaler.transfs_x[iftx]
    sel_tf.steps[0][1].func = dequant
    sel_tf.steps[0][1].inverse_func = requant

    sel_tf.steps[2][1].func = scipy.special.logit
    sel_tf.steps[2][1].inverse_func = scipy.special.expit


fpos = np.arange(0, 100_000, step)
for i in tqdm(fpos):
    pcs = read_chunks([(file_manager.files[0], i, i + step)])
    gl = []
    for pc in pcs:
        graph = contruct_graph_from_row(pc)
        graph.n_pointsv = (
            graph.y[..., conf.loader.y_features.index("num_particles")]
            .int()
            .reshape(-1)
        )

        gl.append(graph)

    batch = Batch.from_data_list(gl)
    batch_untf = batch.clone()
    batch.y = scaler.transform(batch.y, "y")

    ## Manual scaling
    # maxscaler = 1
    maxscaler = 5
    x = batch.x.clone()
    # forward
    for iftx in range(4):
        ix = x[:, [iftx]].double().numpy()
        sel_tf = scaler.transfs_x[iftx]
        tfsteps = [e[1] for e in sel_tf.steps][: maxscaler + 1]

        for itf in tfsteps:
            x_tf = itf.transform(ix)
            x_tf_inv = itf.inverse_transform(x_tf)
            assert np.allclose(x_tf_inv, ix)
            ix = x_tf
        x[:, [iftx]] = torch.tensor(ix).float()

    x_forward = x.clone()

    # backward
    for iftx in range(4):
        ix = x[:, [iftx]].double().numpy()
        sel_tf = scaler.transfs_x[iftx]
        tfsteps = [e[1] for e in sel_tf.steps][: maxscaler + 1]

        for itf in tfsteps[::-1]:
            x_tf = itf.inverse_transform(ix)
            x_tf_inv = itf.transform(x_tf)

            if isinstance(itf, FunctionTransformer) and itf.func == dequant:
                assert np.allclose(x_tf_inv.astype("int"), ix.astype("int"))
            else:
                assert np.allclose(x_tf_inv, ix)
            ix = x_tf
        x[:, [iftx]] = torch.tensor(ix).float()

    assert torch.allclose(x, batch_untf.x)

    # batch.x = x
    batch.y = scaler.inverse_transform(batch.y, "y")

    ## Scaler scaling
    batch.x = scaler.transform(batch_untf.x.clone(), "x")
    batch_scaled = batch.clone()
    _inv = scaler.inverse_transform(batch.x, "x")
    assert torch.allclose(_inv, x)
    assert torch.allclose(_inv, batch_untf.x)
    batch.x = _inv

    # alphapos = conf.loader.x_features.index("alpha")
    # batch.x[..., alphapos] = rotate_alpha(
    #     batch.x[..., alphapos].clone(), batch.batch, True
    # )

    delta = (batch.x - batch_untf.x).abs()
    idx = torch.where(delta == delta.max())

    if not torch.allclose(batch.x, batch_untf.x, rtol=1e-04):
        sel_tf = scaler.transfs_x[idx[1]]
        print(batch.x[idx], batch_untf.x[idx], idx[1])
        print(
            batch_scaled.x[idx],
            sel_tf.transform(batch_untf.x[:, idx[1]]).squeeze()[idx[0]],
            idx[1],
        )

        tfsteps = [e[1] for e in sel_tf.steps]
        forward_res_step = [batch_untf.x[idx].numpy().reshape(-1, 1)]
        for _tf in tfsteps:
            forward_res_step.append(_tf.transform(forward_res_step[-1].copy()))

        backward_res_step = [batch_scaled.x[idx].numpy().reshape(-1, 1)]
        print(f"tranformed {backward_res_step[-1]}, scaled {forward_res_step[-1]}")
        for itf, _tf in enumerate(tfsteps[::-1]):
            backward_res_step.append(
                _tf.inverse_transform(backward_res_step[-1].copy())
            )
            a = backward_res_step[-1].squeeze()
            b = forward_res_step[-2 - itf].squeeze()
            print(a, b, "out of", _tf)

        raise Exception()

    assert torch.allclose(batch.x, batch_untf.x, rtol=1e-04)

    ## cc postprocess step by step

    # alphashift
    # alphas = batch.x[..., alphapos].clone()
    # shift = torch.randint(0, num_alpha,
    #  (batch.batch[-1] + 1,)).to(alphas.device)[
    #     batch.batch
    # ]
    # alphas = alphas.clone() + shift.float()
    # alphas[alphas > num_alpha - 1] -= num_alpha
    # batch.x[..., alphapos] = alphas

    batch = sum_multi_hits(batch, True)
    assert torch.allclose(batch.x, batch_untf.x)

    # batch = batch_to_Exyz(batch)
    # metrics: list[str] = conf.training.val.metrics
    # if "sphereratio" in metrics:
    #     batch["sphereratio"] = sphereratio(batch)
    # if "fpc" in metrics:
    #     batch["fpc"] = fpc_from_batch(batch)
    # if "showershape" in metrics:
    #     batch["showershape"] = analyze_layers(batch)
    # if "response" in metrics:
    #     batch["response"] = response(batch)

    assert torch.allclose(batch.x, batch_untf.x)
    pcs_out = pc_to_voxel(batch)
    input_images = torch.stack([e[1].reshape(*dims) for e in pcs], 0)
    assert torch.allclose(pcs_out.float(), input_images)
