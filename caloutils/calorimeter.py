from typing import Optional


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
    def num_r(self):
        self._assert_calo_init()
        return self._num_r

    @property
    def alppha(self):
        self._assert_calo_init()
        return self._num_r

    @property
    def z(self):
        self._assert_calo_init()
        return self._num_r

    @property
    def dims(self):
        self._assert_calo_init()
        return self._num_r

    def set_layout_calochallange_ds2(self):
        self._num_z, self._num_alpha, self._num_r = self._ds2
        self._dims = self._ds2
        self._caloname = "cc_ds2"

    def set_layout_calochallange_ds3(self):
        self._num_z, self._num_alpha, self._num_r = self._ds3
        self._dims = self._ds3
        self._caloname = "cc_ds3"


calorimeter = Calorimeter()
