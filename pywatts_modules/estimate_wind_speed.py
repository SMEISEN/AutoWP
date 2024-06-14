import numpy as np
import xarray as xr
from typing import Dict

from pywatts.core.base import BaseEstimator


class EstimateWindSpeed(BaseEstimator):
    """
    Estimator to transform wind speed forecasts at 100 m above ground to hub height using the wind profile power law
    """

    def __init__(self, hub_height: float = None, alpha: float = None, name: str = "EstimateWindSpeed"):
        """
        Initialize the step.
        :param hub_height:
            Hub height of the Wind Power (WP) turbine (optional, default=None).
            If not provided, the hub height is estimated.
        :type hub_height: float
        :param alpha:
            Hellman exponent of the wind profile power law (optional, default=1/7).
            Recommended values for onshore = 1/7 and offshore = 1/9
        :type alpha: float
        :param name:
            Step name in the pyWATTS pipeline (optional, default='EstimateWindSpeed').
        :type name: str
        """

        super().__init__(name)

        self.hub_height = hub_height
        self.alpha = alpha

        self.alpha_ = None
        self.hub_height_ = None

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the EstimateWindSpeed object.
        :return:
            Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "hub_height": self.hub_height,
            "alpha": self.alpha
        }

    def set_params(self, hub_height: float = None, alpha: float = None) -> None:
        """
        Set or change EstimateWindSpeed object parameters.
        :param hub_height:
            Hub height of the Wind Power (WP) turbine (optional, default=None).
            If not provided, the hub height is estimated based on the given Hellmann exponent.
        :type hub_height: float
        :param alpha:
            Hellman exponent of the wind profile power law (optional, default=None).
            Recommended default values for onshore = 1/7 and offshore = 1/9.
            If not provided, the Hellmann exponent is estimated based on the given height.
        :type alpha: float
        :return: None
        :rtype: None
        """
        if hub_height is None and alpha is None:
            raise SyntaxError("One of 'hub_height' or 'alpha' must be given!")

        if hub_height is not None:
            self.hub_height = hub_height
        if alpha is not None:
            self.alpha = alpha

    def fit(self, U100: xr.DataArray, V100: xr.DataArray, target: xr.DataArray) -> None:
        """
        Fit the module.
        :param U100:
            Wind speed at 100 m above ground (x-axis)
        :type U100: xr.DataArray
        :param V100:
            Wind speed at 100 m above ground (y-axis)
        :type V100: xr.DataArray
        :param target:
            Wind speed measurement at hub height (total).
        :type target: xr.DataArray
        :return: None
        :rtype: None
        """

        # Calculate total wind speed at 100 m above ground
        wind_speed_100 = (U100 ** 2 + V100 ** 2) ** (1 / 2)

        if self.alpha is not None and self.hub_height is None:
            # Estimate hub_height based on given alpha
            self.alpha_ = self.alpha
            self.hub_height_ = np.nanmedian(100 * (target / wind_speed_100) ** (1 / self.alpha_))
        elif self.alpha is None and self.hub_height is not None:
            # Estimate alpha based on given hub_height
            self.hub_height_ = self.hub_height
            self.alpha_ = np.nanmedian(np.log(target / wind_speed_100) / np.log(self.hub_height_ / 100))
        else:
            # Use given hub height and given alpha
            self.hub_height_ = self.hub_height
            self.alpha_ = self.alpha

        self.is_fitted = True

    def transform(self, U100, V100) -> xr.DataArray:
        """
        Estimate wind speed at hub height based on estimated hub height or estimated Hellmann exponent.
        :param U100:
            Wind speed at 100 m above ground (x-axis)
        :type U100: xr.DataArray
        :param V100:
            Wind speed at 100 m above ground (y-axis)
        :type V100: xr.DataArray
        :return:
            Effective wind speed.
        :rtype: xr.DataArray
        """

        # Calculate total wind speed at 100 m above ground
        wind_speed_100 = (U100 ** 2 + V100 ** 2) ** (1 / 2)

        # Calculate effective wind speed at hub height
        wind_speed_hub = wind_speed_100 * (self.hub_height_ / 100) ** self.alpha_

        return wind_speed_hub
