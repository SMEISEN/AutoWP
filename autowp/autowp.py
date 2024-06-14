import warnings
import pandas as pd
from typing import Union, Dict

from autowp.sub_pipelines.ensembling import add_ensemble
from autowp.sub_pipelines.preprocessing import add_preprocessing
from autowp.sub_pipelines.model_pool import add_models
from autowp.sub_pipelines.utils import create_modules, save_modules, load_modules, load_power_curves
from autowp.config import MeasurementUnit, AdaptionConfiguration
from pywatts.core.pipeline import Pipeline
from pywatts.core.summary_formatter import SummaryJSON


class AutoWP:
    """
    AutoWP: Template for automated wind power forecasts
    The underlying idea of AutoWP is to represent a new Wind Power (WP) turbine as a convex linear combination of WP
    curves from a sufficiently diverse ensemble. The method consists of three steps: i) create the ensemble of normalized
    WP curves, ii) form the normalized ensemble WP curve by the optimally weighted sum of the WP curves in the ensemble,
    and iii) re-scale the ensemble WP curve with the new WP turbine's peak power rating.
    """
    def __init__(self, target_power_generation: str,
                 measurement_unit: MeasurementUnit, samples_per_hour: int = None,
                 target_kWp: float = None,
                 target_wind_speed_hub: str = None,
                 power_curves: Dict[str, pd.DataFrame] = None,
                 wpl_related: bool = False, wpl_tolerance: float = 0.05, wpl_normalized: bool = True, wpl_limit: int = 10,
                 adaption_config: AdaptionConfiguration = AdaptionConfiguration.none,
                 C: Union[pd.Timedelta, None] = None, K: Union[pd.Timedelta, None] = None):
        """
        :param target_power_generation:
            Column name of the target WP turbine in the dataframe given in train and predict.
        :type target_power_generation: str
        :param measurement_unit:
            Unit of the WP generation, can be kW or kWp.
        :type measurement_unit: MeasurementUnit
        :param samples_per_hour:
            Resolution of the time series in samples per hour (optional, default=None).
            Required for transforming an energy metering time series into the mean power time series.
        :type samples_per_hour: int
        :param target_kWp:
            Peak power rating of the target WP turbine (optional, default=None).
            If not provided, it is estimated based on the unconstrained sum of ensemble weights.
        :type target_kWp: float
        :param target_wind_speed_hub:
            Column name of the wind speed measurement at hub height of the target WP turbine in the dataframe given in train
            and predict (optional, default=None).
            If not provided, it is recommended to let AutoWP estimate the peak power of the WP turbine to implicitly consider
            the height correction in the ensemble optimization.
        :type target_wind_speed_hub: str
        :param power_curves:
            WP curves for the ensemble optimization (optional, default=None).
            If not provided, a diverse ensemble of WP curves from the wind power lib are used.
        :type power_curves: dict(key: WP turbine name, value: pd.Dataframe(index: wind speed, column: power))
        :param wpl_related:
            Weather to search for WP curves with similar peak power rating in the wind power lib (optional, default=False).
            If False, all WP curves are considered (requires wpl_normalized=True).
        :type wpl_related: bool
        :param wpl_tolerance:
            Tolerance for searching WP curves with similar peak power rating in the wind power lib (optional, default=0.05).
        :type wpl_tolerance: str
        :param wpl_normalized:
            Weather to normalize the WP curves in the wind power lib (optional, default=True).
        :type wpl_normalized: bool
        :param wpl_limit:
            Number of WP curves from the wind power lib to be considered in the ensemble (optional, default=10).
            For efficient computation, AutoWP selects a diverse set of WP curves to reduce while preserving diversity.
        :type wpl_limit: int
        :param adaption_config:
            Configuration for the online adaption of ensemble weights (optional, default=AdaptionConfiguration.None).
        :type adaption_config: AdaptionConfiguration
        :param C:
            Adaption cycle for the online adaption of ensemble weights (optional, default=None).
        :type C: pd.Timedelta
        :param K:
            Adaption batch (past horizon) for the online adaption of ensemble weights (optional, default=None).
        :type K: pd.Timedelta
        """

        if adaption_config != AdaptionConfiguration.none and (C is None or K is None):
            raise SyntaxError("The adaption cycle C and the number of considered samples K must be given!")

        if measurement_unit == MeasurementUnit.kWh and samples_per_hour is None:
            raise SyntaxError("The measurement unit is kWh. Transforming the energy metering data into a mean power "
                              "generation time series requires the number of samples per hour!")
        if wpl_related and target_kWp is None:
            raise SyntaxError("Cannot select related wind power curves if the target wind turbine's peak power rating"
                              "is not given!")
        if not wpl_related and not wpl_normalized:
            raise SyntaxError("Considering all WP curves in the wind power lib requires normalizing the curves!")

        if target_kWp is not None and target_wind_speed_hub is None:
            warnings.warn("The wind power turbine's peak power rating is given while the effective wind speed measurement at"
                          "hub height is not given. In order to increase the adaptability of the wind power curve ensemble,"
                          "it is recommended to let the peak power rating be estimated, i.e. target_kWp = None.")

        self.target_power_generation = target_power_generation
        self.target_wind_speed_hub = target_wind_speed_hub

        self.target_kWp = target_kWp

        self.wpl_related = wpl_related
        self.wpl_tolerance = wpl_tolerance
        self.wpl_normalized = wpl_normalized
        self.wpl_limit = wpl_limit

        self.measurement_unit = measurement_unit
        self.samples_per_hour = samples_per_hour

        self.adaption_config = adaption_config
        self.C = C
        self.K = K

        if power_curves is not None:
            self._power_curves = power_curves
        else:
            self._power_curves, _, _, _ = load_power_curves(target_kWp=self.target_kWp, tolerance=self.wpl_tolerance,
                                                            limit=self.wpl_limit, related=self.wpl_related,
                                                            normalized=self.wpl_normalized)

        self.weights_ = None
        self.kWp_ = target_kWp if target_kWp is not None else None
        self._modules = create_modules(power_curves=self._power_curves, target_kWp=self.target_kWp)
        self._create_pipeline()

    def _estimate_peak_power_rating(self) -> float:
        """
        Estimates the WP turbine's peak power rating from the unconstrained sum of ensemble weights.
        :return:
            Estimated peak power rating of the WP turbine in kilo Watt (peak).
        :rtype: float
        """

        if self.measurement_unit == MeasurementUnit.kW:
            return sum(self.weights_)
        elif self.measurement_unit == MeasurementUnit.kWh:
            # samples_per_hour: convert the energy metering time series into the mean power time series
            return sum(self.weights_) * self.samples_per_hour

    def _create_pipeline(self) -> None:
        """
        Creates the pyWATTS pipeline, consisting of pre-processing, WP curve model creation, and forecast ensembling.
        :return: None
        :rtype: None
        """

        self._pipeline = Pipeline(path=f"../results/pipeline", name="AutoWP", batch=self.C)
        self._pipeline = add_preprocessing(pipeline=self._pipeline, modules=self._modules,
                                           target_power_generation=self.target_power_generation,
                                           target_wind_speed_hub=self.target_wind_speed_hub,
                                           measurement_unit=self.measurement_unit,
                                           samples_per_hour=self.samples_per_hour)
        self._pipeline = add_models(pipeline=self._pipeline, modules=self._modules, power_curves=self._power_curves)
        self._pipeline = add_ensemble(pipeline=self._pipeline, modules=self._modules,
                                      target_power_generation=self.target_power_generation,
                                      adaption_config=self.adaption_config, K=self.K)

    def fit(self, data: pd.DataFrame,
            save_models: bool = False, model_directory: str = "../models") -> dict:
        """
        Fit AutoWP to the data.
        :param data:
            Data consisting of the wind speed forecast (U100 and V100) and the target WP turbine's generation as specified
            in the initialization with target_power_generation
        :type data: pd.DataFrame
        :param save_models:
            Weather to save the AutoWP model as pickle-file (optional, default=False)
        :type save_models: bool
        :param model_directory:
            Directory of saved AutoWP model (optional, default="../models")
        :type model_directory: str
        :return:
            Predictions on the training data. Consists of the effective wind speed 'wind_speed_eff', the WP curve's outputs
            'WPC_<WP turbine manufacturer and WP type>', the ensemble output 'y_hat', and the ground truth 'y'
        :rtype: dict
        """

        res = {}
        if self.adaption_config == AdaptionConfiguration.none:
            # Train pipeline
            res = self._pipeline.train(data=data, summary=False, summary_formatter=SummaryJSON())

            # Get weights from pipeline
            self.weights_ = self._modules["ensemble"].weights_

            # Estimate peak power rating
            if self.kWp_ is None:
                self.kWp_ = self._estimate_peak_power_rating()

        # Save modules of pipeline
        if save_models:
            save_modules(pipeline_modules=self._modules, dry=model_directory)

        return res

    def predict(self, data: pd.DataFrame, online_start: pd.Timestamp = None,
                load_models: bool = False, model_directory: str = "../models") -> dict:
        """
        Predict using the AutoWP model.
        :param data:
            Data consisting of the wind speed forecast (U100 and V100) and the target WP turbine's generation as specified
            in the initialization with target_power_generation
        :type data: pd.DataFrame
        :param online_start:
            Time point where the online simulation should start (optional, default=None).
            The time point must be within the index of the given data.
        :type online_start: pd.Timestamp
        :param load_models:
            Weather to load the AutoWP model from a pickle-file (optional, default=False)
        :type load_models: bool
        :param model_directory:
            Directory of saved models (optional, default="../models")
        :type model_directory: str
        :return:
            Predictions on the training data. Consists of the effective wind speed 'wind_speed_eff', the WP curve's outputs
            'WPC_<WP turbine manufacturer and WP type>', the ensemble output 'y_hat', and the ground truth 'y'
        :rtype: dict
        """

        if self.C is None and online_start is not None:
            warnings.warn("The adaption cycle C is None. The argument online_start will be ignored!")
        if self.C is not None and online_start is None:
            raise SyntaxError("The online simulation requires defining the timestamp where the online processing begins!")

        # Load modules of pipeline
        if load_models:
            self._modules = load_modules(pipeline_modules=self._modules, dry=model_directory)
            self._create_pipeline()

        # Predict with pipeline, online simulation if online_start is not None
        result = self._pipeline.test(data=data, summary=False, summary_formatter=SummaryJSON(), online_start=online_start)

        # Get weights from pipeline
        self.weights_ = self._modules["ensemble"].weights_

        # Estimate peak power rating
        if self.kWp_ is None:
            self.kWp_ = self._estimate_peak_power_rating()

        return result
