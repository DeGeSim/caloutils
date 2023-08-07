from math import prod
from typing import Optional

import torch


class Calorimeter:
    def __init__(self) -> None:
        self._dims: Optional[tuple[int, int, int]] = None
        self._num_r: Optional[int] = None
        self._num_alpha: Optional[int] = None
        self._num_z: Optional[int] = None
        self._caloname: Optional[str] = None

    def _assert_calo_init(self):
        if self._caloname is None:
            raise Exception(
                "Calorimeter is not initalized. Use"
                " `init_calorimeter(caloname)` before accassing the attributes."
            )

    @property
    def cell_idxs(self) -> torch.Tensor:
        return torch.arange(prod(self.dims)).reshape(*(self.dims))

    @property
    def num_r(self) -> int:
        self._assert_calo_init()
        return self._num_r

    @property
    def num_alpha(self) -> int:
        self._assert_calo_init()
        return self._num_alpha

    @property
    def num_z(self) -> int:
        self._assert_calo_init()
        return self._num_z

    @property
    def dims(self) -> tuple[int, int, int]:
        self._assert_calo_init()
        return self._dims

    def pos_to_cellidx(self, pos: torch.Tensor):
        assert pos.shape[-1] == len(self.dims)
        dev = pos.device
        return self.cell_idxs.to(dev)[pos[..., 0], pos[..., 1], pos[..., 2]]

    def init_calorimeter(self, caloname: str):
        """The function `init_calorimeter` initializes a calorimeter geometry.
        Currently implemented are dataset 2 and 3 of the CaloChallenge:
        https://github.com/CaloChallenge/homepage

        Parameters
        ----------
        caloname : str
            The parameter `caloname` is a string that represents the name of the calorimeter. It can have two
        possible values: "cc_ds2" or "cc_ds3".

        """
        match caloname:
            case "cc_ds2":
                layout = (45, 16, 9)
            case "cc_ds3":
                layout = (45, 50, 18)
            case "test":
                layout = (2, 2, 1)
            case _:
                raise NotImplementedError(
                    f"No such calorimeter: {caloname}. Options are"
                    " :'cc_ds2','cc_ds3'"
                )
        self._caloname = caloname
        self._num_z, self._num_alpha, self._num_r = layout
        self._dims = layout


calorimeter = Calorimeter()
