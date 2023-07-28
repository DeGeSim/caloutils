import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.validation import FLOAT_DTYPES


class IdxToScale(MinMaxScaler):
    """Same as MinMaxScaler but fits min to the floor and Max to the roof"""

    # _parameter_constraints: dict = {
    #     "feature_range": [tuple],
    #     "copy": ["boolean"],
    #     "clip": ["boolean"],
    # }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def partial_fit(self, X, y=None):
        self._validate_params()

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum."
                " Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        # Change here
        data_min = np.floor(np.nanmin(X, axis=0))
        data_max = np.ceil(np.nanmax(X, axis=0))
        # Change end

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (
            feature_range[1] - feature_range[0]
        ) / _handle_zeros_in_scale(data_range, copy=True)
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self
