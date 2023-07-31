from typing import Optional


class Calorimeter:
    def __init__(self) -> None:
        self._ds2 = (45, 16, 9)
        self._ds3 = (45, 50, 18)
        self.dims: Optional[tuple[int, int, int]] = None
        self.num_z: Optional[int] = None
        self.num_alpha: Optional[int] = None
        self.num_r: Optional[int] = None

    def set_layout_calochallange_ds2(self):
        self.num_z, self.num_alpha, self.num_r = self._ds2
        self.dims = self._ds2

    def set_layout_calochallange_ds3(self):
        self.num_z, self.num_alpha, self.num_r = self._ds3
        self.dims = self._ds3


calorimeter = Calorimeter()
