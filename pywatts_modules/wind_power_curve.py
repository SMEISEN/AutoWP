from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from scipy.interpolate import interp1d


class WindPowerCurveTransformer(BaseTransformer):
    """
    Transformer to estimate the Wind Power (WP) generation based on the wind speed using a WP curve.
    """

    def __init__(self, power_curve: pd.DataFrame,
                 wind_speed_activation: float, wind_speed_nominal: float, wind_speed_shutdown: float,
                 kWp: float = None, interp_kind: str = 'cubic', name: str = "WPC"):
        """
        Initialize the step.
        :param power_curve:
            WP curve.
        :type power_curve: pd.Dataframe(index: wind speed, column: power)
        :param wind_speed_activation:
            Cut-in weind speed.
        :type wind_speed_activation: float
        :param wind_speed_nominal:
            Wind speed at rated power.
        :type wind_speed_nominal: float
        :param wind_speed_shutdown:
            Cut-out wind speed.
        :type wind_speed_shutdown: float
        :param kWp:
            Peak power rating of the WP turbine.
        :type kWp: float
        :param interp_kind:
            Kind of interpolation used (optional, default='cubic').
        :type interp_kind: str
        :param name:
            Step name in the pyWATTS pipeline (optional, default='WPC')
        :type name: str
        """

        super().__init__(name)
        self.power_curve = power_curve
        self.kWp = kWp
        self.wind_speed_activation = wind_speed_activation
        self.wind_speed_nominal = wind_speed_nominal
        self.wind_speed_shutdown = wind_speed_shutdown
        self.interp_kind = interp_kind

        x = self.power_curve["power"].index.values
        y = self.power_curve["power"].values
        if self.kWp is not None:
            # Scale WP curve by peak power rating
            y = np.squeeze(MinMaxScaler(feature_range=(0, self.kWp)).fit_transform(y.reshape(-1, 1)))

        self._sector_1 = lambda _: 0
        self._sector_2 = interp1d(x, y, kind=self.interp_kind)
        self._sector_3 = lambda _: max(y)
        self._sector_4 = lambda _: 0

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the WindPowerCurveTransformer object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "power_curve": self.power_curve,
            "wind_speed_activation": self.wind_speed_activation,
            "wind_speed_nominal": self.wind_speed_nominal,
            "wind_speed_shutdown": self.wind_speed_shutdown,
            "interp_kind": self.interp_kind
        }

    def set_params(self, power_curve: pd.DataFrame = None,
                   wind_speed_activation: float = None, wind_speed_nominal: float = None,
                   wind_speed_shutdown: float = None, kWp: float = None, interp_kind: str = None
                   ) -> None:
        """
        Set or change parameters of the WindPowerCurveTransformer object.
        :param power_curve:
            WP curve.
        :type power_curve: pd.Dataframe(index: wind speed, column: power)
        :param wind_speed_activation:
            Cut-in weind speed.
        :type wind_speed_activation: float
        :param wind_speed_nominal:
            Wind speed at rated power.
        :type wind_speed_nominal: float
        :param wind_speed_shutdown:
            Cut-out wind speed.
        :type wind_speed_shutdown: float
        :param kWp:
            Peak power rating of the WP turbine.
        :type kWp: float
        :param interp_kind:
            Kind of interpolation used (optional, default='cubic').
        :type interp_kind: str
        :return: None
        :rtype: None
        """
        if power_curve is not None:
            self.power_curve = power_curve
        if wind_speed_activation is not None:
            self.wind_speed_activation = wind_speed_activation
        if wind_speed_nominal is not None:
            self.wind_speed_nominal = wind_speed_nominal
        if wind_speed_shutdown is not None:
            self.wind_speed_shutdown = wind_speed_shutdown
        if kWp is not None:
            self.kWp = kWp
        if interp_kind is not None:
            self.interp_kind = interp_kind

    def transform(self, wind_speed) -> xr.DataArray:
        """
        Forecast the wind power generation for the given input.

        :param wind_speed:
            Wind speed at hub height.
        :type wind_speed: xr.DataArray
        :return:
            Prediction.
        :rtype: xr.DataArray
        """

        return numpy_to_xarray(self.wind_power_curve(wind_speed.values), wind_speed)

    def wind_power_curve(self, wind_speed: np.array) -> np.piecewise:
        """
        Piecewise wind power curve function.
        :param wind_speed:
            Wind speed input for the wind power curve.
        :type wind_speed: np.array
        :return:
            Wrapped scipy interpolator.
        :rtype: np.piecewise
        """

        piecewise = np.piecewise(wind_speed, [
            wind_speed < self.wind_speed_activation,
            ((wind_speed >= self.wind_speed_activation) & (wind_speed < self.wind_speed_nominal)),
            ((wind_speed >= self.wind_speed_nominal) & (wind_speed < self.wind_speed_shutdown)),
            wind_speed >= self.wind_speed_shutdown
        ], [
                                     lambda ws: self._sector_1(ws),
                                     lambda ws: self._sector_2(ws),
                                     lambda ws: self._sector_3(ws),
                                     lambda ws: self._sector_4(ws)
                                 ])

        return piecewise
