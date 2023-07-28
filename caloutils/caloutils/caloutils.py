"""Main module."""
import torch
from torch_geometric.data import Batch
from preprocessing.voxelize import sum_duplicate_hits, voxelize
from preprocessing.convcoord import batch_to_Exyz
from preprocessing.pca import fpc_from_batch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.data import Data
from torch_scatter import scatter_std
class particle_cloud_evaluation():
    def __init__(self,name,detector_shape=None, metrics_to_calculate=[]) -> None:
        self.name=name
        self.num_z,self.num_alpha,self.num_R = 45,16,9 if name=="dataset_2" else 45,50,18 if name=="dataset_3" else detector_shape
        self.vars_to_calculate = metrics_to_calculate

    def convert_to_pc(self, Einc,voxel) -> Batch:
        E, shower = Einc, voxel
        shower = shower.reshape(self.num_z, self.num_alpha, self.num_r)
        idxs = torch.where(shower)
        h_energy = shower[idxs]
        z, alpha, r = idxs
        pc = torch.stack([h_energy, z, alpha, r]).T
        assert not pc.isnan().any()
        num_hits = torch.tensor(idxs[0].shape).float()
        batch=self.convert_to_batch(pc,Einc,num_hits)
        return batch

    def convert_to_batch(self,pc,Einc,num_hits) -> Batch:
        E, pc= pc[...,0],pc[...,1:]
        batch = Data(x=pc, y=torch.concat([E, num_hits, E / num_hits]).reshape(1, 3)) # TODO Why the third column?
        return batch

    def postprocess(self,batch: Batch, sim_or_gen: str) -> Batch:
        if sim_or_gen == "gen":
            alphas = (batch.x[..., 2] + torch.randint(0, self.num_alpha, (len(batch.x[...,2]), 1,)).to(
                alphas.device
            )[batch.batch]) % self.num_alpha
            batch.x[..., 2] = alphas
            return batch

    def analyze_layers(self,batch: Batch) -> dict[str, torch.Tensor]:
        """Get the ratio between half-turnon and
        half turnoff to peak center from a PC"""
        batchidx = batch.batch
        n_events = int(batchidx[-1] + 1)
        device = batch.x.device
        z = batch.xyz[..., -1]
        # compute the histogramm with dim (events,layers)
        z_ev = (z.long() + batchidx * self.num_z).sort().values

        zvals, zcounts = z_ev.unique_consecutive(return_counts=True)

        layer_idx = zvals % self.num_z
        ev_idx = (zvals - layer_idx) // self.num_z

        hist = torch.zeros(n_events, self.num_z).long().to(device)
        hist[ev_idx, layer_idx] = zcounts

        assert (hist.sum(1) == batch.ptr[1:] - batch.ptr[:-1]).all()

        max_hits_per_event, peak_layer_per_event = hist.max(1)

        occupied_layers = (
            hist > max_hits_per_event.reshape(-1, 1).repeat(1, self.num_z) / 2
        )
        eventidx, occ_layer = occupied_layers.nonzero().T
        # find the min and max layer from the bool matrix ath the same time
        minmax = global_max_pool(torch.stack([occ_layer, -occ_layer]).T, eventidx)
        # reverse the -1 trick
        turnoff_layer, turnon_layer = (minmax * torch.tensor([1, -1]).to(device)).T

        # distance to the peak
        turnoff_dist = turnoff_layer - peak_layer_per_event
        turnon_dist = peak_layer_per_event - turnon_layer
        psr = (turnoff_dist + 1) / (turnon_dist + 1)

        return {
            "peak_layer": peak_layer_per_event.float(),
            "psr": psr.float(),
            "turnon_layer": turnon_layer.float(),
        }


    def sphereratio(self,batch: Batch) -> dict[str, torch.Tensor]:
        batchidx = batch.batch
        # Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)
        Ehit = batch.x[:, 0].reshape(-1, 1)
        e_small, e_large = self.__dist_fraction(Ehit, batch.xyz, batchidx, 0.3, 0.8)

        return {
            "small": e_small,
            "large": e_large,
            "ratio": e_small / e_large,
        }


    def cyratio(self,batch: Batch) -> dict[str, torch.Tensor]:
        batchidx = batch.batch
        # Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)
        Ehit = batch.x[:, 0].reshape(-1, 1)

        e_small, e_large = self.__dist_fraction(
            Ehit, batch.xyz[:, [0, 1]], batchidx, 0.2, 0.6
        )

        return {
            "small": e_small,
            "large": e_large,
            "ratio": e_small / e_large,
        }


    def __dist_fraction(self,Ehit, pos, batchidx, small, large, center_energy_weighted=True):
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



    def response(self,batch: Batch) -> torch.Tensor:
        batchidx = batch.batch
        # Ehit = batch.x[:, conf.loader.x_ftx_energy_pos]
        Ehit = batch.x[:, 0]
        Esum = global_add_pool(Ehit, batchidx)
        return Esum / batch.y[:, 0]

    def calc_vars(self,batch:Batch,fake=False):
        if fake:
            batch = sum_duplicate_hits(batch)
        batch = batch_to_Exyz(batch)
        if "sphereratio" in self.vars:
            batch["sphereratio"] = self.sphereratio(batch)
        if "cyratio" in self.vars:
            batch["cyratio"] = self.cyratio(batch)
        if "fpc" in self.vars:
            batch["fpc"] = self.fpc_from_batch(batch)
        if "showershape" in self.vars:
            batch["showershape"] = self.analyze_layers(batch)
        if "response" in self.vars:
            batch["response"] = self.response(batch)

        return batch

    def calc_metrics(self, real, fake):
        metrics = {}
        for var in self.vars:
            metrics[var] = (torch.abs(fake[var].values().cumsum()-real[var].values().cumsum())/fake[var].values()[-1]).mean()
        return metrics

