from math import prod
from typing import Optional

import torch


class Calorimeter:
    def __init__(self) -> None:
        self._ds2 = (45, 16, 9)
        self._ds3 = (45, 50, 18)
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

    def set_layout_calochallange_ds2(self):
        self._num_z, self._num_alpha, self._num_r = self._ds2
        self._dims = self._ds2
        self._caloname = "cc_ds2"

    def set_layout_calochallange_ds3(self):
        self._num_z, self._num_alpha, self._num_r = self._ds3
        self._dims = self._ds3
        self._caloname = "cc_ds3"


calorimeter = Calorimeter()
