from typing import Union, List, Dict
from enum import IntEnum

import logging
import xarray as xr
import numpy as np

from scipy.optimize import least_squares

from pywatts.core.base import BaseEstimator
from pywatts.core.exceptions import WrongParameterException
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.utils._split_kwargs import split_kwargs

logger = logging.getLogger(__name__)


class LossMetric(IntEnum):
    """
    Enum which contains the different loss metrics of the ensemble module.
    MSE: Mean Squared Error (MSE)
    MAE: Mean Absolute Error (MAE)
    """
    MSE = 1
    MAE = 2


class EnsembleFunction(IntEnum):
    """
    Enum which contains the different ensemble optimization functions of the ensemble module.
    WeightedSum: Weighted sum without constraining the weights to be in [0,1] and sum up to 1.
    WeightedAverage: Weighted averaging with constraining the weights to be in [0,1] and sum up to 1.
    """
    WeightedSum = 1
    WeightedAverage = 2


class Ensemble(BaseEstimator):
    """
    Aggregation estimator to ensemble the given input time series.
    """

    def __init__(self, weights: Union[str, list] = None, k_best: Union[str, int] = None,
                 inputs_ignored: List[str] = None,
                 ensemble_fun: EnsembleFunction = EnsembleFunction.WeightedAverage, loss_metric: LossMetric = LossMetric.MSE,
                 max_weight: float = 1.0, name: str = "Ensemble"):
        """
        Initialize the step.
        :param weights:
            Method for weighting the given input time series (optional, default=None).
            None, simple averaging
            list, of given weights
            'autoErr', weights based on in-sample loss
            'autoOpt', weights based on ensemble optimization
        :type weights: Union[str, list]
        :param k_best:
            Weather to drop forecasts depending on in-sample loss and keep the k-best forecasts if the 'autoErr' strategy is
            chosen (optional, default=None).
            'autoErr', automatically drop forecasts with high in-sample loss
            int, keep the k-best forecasts based on in-sample loss
        :type k_best: Union[str, int]
        :param inputs_ignored:
            List of input names to be ignored in the ensemble, e.g., when testing with a leave-one-out-evaluation
            (optional, default=None).
        :type inputs_ignored: List[str]
        :param ensemble_fun:
            Selection of the implemented objective functions in the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=EnsembleFunction.WeightedAverage)
        :type ensemble_fun: EnsembleFunction
        :param loss_metric:
            Loss metric used for the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=LossMetric.MSE).
        :type loss_metric: LossMetric
        :param max_weight:
            Defines the maximum weight of one forecast in the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=1.0).
        :type max_weight: float
        :param name:
            Step name in the pyWATTS pipeline (optional, default='Ensemble')
        :type name: str
        """

        super().__init__(name)
        if inputs_ignored is None:
            inputs_ignored = []

        self.weights = weights
        self.k_best = k_best
        self.inputs_ignored = inputs_ignored
        self.max_weight = max_weight
        self.ensemble_fun = ensemble_fun
        self.loss_metric = loss_metric

        self.weights_ = None
        self.weights_autoOpt_ = None

        self.is_fitted = False

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the Ensemble object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "weights": self.weights,
            "k_best": self.k_best,
            "inputs_ignored": self.inputs_ignored,
            "max_weight": self.max_weight,
            "ensemble_fun": self.ensemble_fun,
            "loss_metric": self.loss_metric,
        }

    def set_params(self,
                   weights: Union[str, list] = None,
                   k_best: Union[str, int] = None,
                   inputs_ignored: List[str] = None,
                   max_weight: float = None,
                   ensemble_fun: EnsembleFunction = None,
                   loss_metric: LossMetric = None,
                   ) -> None:
        """
        Set or change parameters of the Ensemble object.
        :param weights:
            Method for weighting the given input time series (optional, default=None).
            None, simple averaging
            list, of given weights
            'autoErr', weights based on in-sample loss
            'autoOpt', weights based on ensemble optimization
        :type weights: Union[str, list]
        :param k_best:
            Weather to drop forecasts depending on in-sample loss and keep the k-best forecasts if the 'autoErr' strategy is
            chosen (optional, default=None).
            'autoErr', automatically drop forecasts with high in-sample loss
            int, keep the k-best forecasts based on in-sample loss
        :type k_best: Union[str, int]
        :param inputs_ignored:
            List of input names to be ignored in the ensemble, e.g., when testing with a leave-one-out-evaluation
            (optional, default=None).
        :type inputs_ignored: List[str]
        :param max_weight:
            Defines the maximum weight of one forecast in the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=1.0).
        :type max_weight: float
        :param ensemble_fun:
            Selection of the implemented objective functions in the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=EnsembleFunction.WeightedAverage)
        :type ensemble_fun: EnsembleFunction
        :param loss_metric:
            Loss metric used for the ensemble optimization if the 'autoOpt' strategy is chosen
            (optional, default=LossMetric.MSE).
        :type loss_metric: LossMetric
        """

        if weights is not None:
            self.weights = weights
        if k_best is not None:
            self.k_best = k_best
        if inputs_ignored is not None:
            self.inputs_ignored = inputs_ignored
        if max_weight is not None:
            self.max_weight = max_weight
        if ensemble_fun is not None:
            self.ensemble_fun = ensemble_fun
        if loss_metric is not None:
            self.loss_metric = loss_metric

    def fit(self, **kwargs) -> None:
        """
        Fit the module.
        :param kwargs:
            Input time series to be aggregated using an ensemble.
        :type kwargs: dict
        :return: None
        :rtype: None
        """

        if len(self.inputs_ignored) > 0:
            # Remove inputs to be ignored in the ensemble
            kwargs = {key: value for key, value in kwargs.items() if key not in self.inputs_ignored}

        forecasts, targets = split_kwargs(kwargs)  # Split inputs into forecasts and targets

        if self.weights == 'autoErr' or self.k_best is not None:
            # Determine weights depending on in-sample loss
            loss_values = self._calculate_loss_forecasts(p_n=forecasts, t=targets)
            # Drop forecasts depending on in-sample loss
            index_loss_dropped = self._drop_forecasts(loss=loss_values)

            # Overwrite weights based on given loss values and set weights of dropped forecasts to zero
            if self.weights == "autoErr":  # Weighted averaging depending on estimated weights
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else 1 / value for i, value in enumerate(loss_values)])
            elif self.weights is None:  # Averaging
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else 1 for i, value in enumerate(loss_values)])
            else:  # Weighted averaging depending on given weights
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else weight for i, (value, weight) in enumerate(zip(loss_values, self.weights))])

        elif self.weights == 'autoOpt':
            # Determine weights based on ensemble optimization
            predictions = [forecast.values for forecast in forecasts.values()]
            if self.max_weight != 1.0:
                raise UserWarning(f"Parameter max_weight={self.max_weight} is ignored for the least squares optimization!")

            bounds = [[0.0 for _ in range(len(forecasts))]] + \
                     [[1.0 for _ in range(len(forecasts))]]  # Define weight constraints

            x0 = [1 / len(forecasts)] * len(forecasts)  # Initial values
            kwargs = {"p_n": np.array(predictions).T, "t": targets["target"]}  # Static inputs for the objective function

            # Define objective function
            if self.ensemble_fun == EnsembleFunction.WeightedSum:
                fun = self._assess_weighted_sum
            elif self.ensemble_fun == EnsembleFunction.WeightedAverage:
                fun = self._assess_weighted_average

            # Least-squares ensemble optimizaiton
            result = least_squares(fun=fun, x0=x0, bounds=bounds, kwargs=kwargs,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if self.ensemble_fun == EnsembleFunction.WeightedSum:
                self.weights_ = list(result.x)
            elif self.ensemble_fun == EnsembleFunction.WeightedAverage:
                self.weights_ = self._normalize_weights(list(result.x))  # Normalize weights to hold sum(result.x) = 1
            self.weights_autoOpt_ = self.weights_

        elif self.weights is not None:
            # Wse given weights
            if isinstance(self.weights, list):
                if len(self.weights) is not len(forecasts):
                    raise WrongParameterException(
                        "The number of the given weights does not match the number of given forecasts.",
                        f"Make sure to pass {len(forecasts)} weights.",
                        self.name
                    )
            self.weights_ = self._normalize_weights(self.weights)

        self.is_fitted = True

    def transform(self, **kwargs) -> xr.DataArray:
        """
        Ensemble the given time series using the given ensemble function and given or estimated weights.
        :param kwargs:
            Input time series to be aggregated using an ensemble.
        :type kwargs: dict
        :return:
            Aggregated time series.
        :rtype: xr.DataArray
        """

        if len(self.inputs_ignored) > 0:
            # Remove inputs to be ignored in the ensemble
            kwargs = {key: value for key, value in kwargs.items() if key not in self.inputs_ignored}

        forecasts, _ = split_kwargs(kwargs)  # Split inputs into forecasts (and targets)

        # Create lists of the ensemble pool models' indexes and data values
        list_of_series = []
        list_of_indexes = []
        for series in forecasts.values():
            list_of_indexes.append(series.indexes)
            list_of_series.append(series.data)

        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for averaging do not match")

        # Calculate ensemble output based on the weights
        if self.ensemble_fun == EnsembleFunction.WeightedSum:
            result = np.dot(self.weights_, np.array(list_of_series))
        elif self.ensemble_fun == EnsembleFunction.WeightedAverage:
            result = np.average(np.array(list_of_series), axis=0, weights=self.weights_)

        return numpy_to_xarray(result, series)

    def _assess_weighted_sum(self, x: np.array, p_n: np.array, t: np.array) -> float:
        """
        Assessment function for the weights of a trial using the weighted sum.
        :param x:
            Variable to be optimized.
        :type x: np.array
        :param p_n:
            Forecasts that are ensembled.
        :type p_n: np.array
        :param t:
            Target values, used to fit the ensemble.
        :type t: np.array
        :return:
            Calculated error between prediction and target.
        :rtype: float
        """

        p = np.dot(p_n, x)

        return self._calculate_error(p=p, t=t)

    def _assess_weighted_average(self, x: np.array, p_n: np.array, t: np.array) -> float:
        """
        Assessment function for the weights of a trial using the weighted average.
        :param x:
            Variable to be optimized.
        :type x: np.array
        :param p_n:
            Forecasts that are ensembled.
        :type p_n: np.array
        :param t:
            Target values, used to fit the ensemble.
        :type t: np.array
        :return:
            Calculated error between prediction and target.
        :rtype: float
        """

        p = np.average(p_n.T, axis=0, weights=x)

        return self._calculate_error(p=p, t=t)

    def _calculate_error(self, p: np.array, t: np.array) -> float:
        """
        Calculate the error between prediction and target
        :param p:
            Prediction
        :type p: np.array
        :param t:
            Target
        :type t: np.array
        :return:
            Calculated error, depends on defined loss metric.
        :rtype: float
        """
        if self.loss_metric == LossMetric.MSE:
            return float(np.mean((p - t) ** 2))
        elif self.loss_metric == LossMetric.MAE:
            return float(np.mean(np.abs((p - t))).ravel())
        else:
            raise WrongParameterException(
                "The specified loss metric is not implemented.",
                "Make sure to pass LossMetric.MSE or LossMetric.MAE.",
                self.name)

    def _calculate_loss_forecasts(self, p_n: dict, t: np.array) -> list:
        """
        Calculates the loss of the given forecasts.
        :param p_n:
            Predictions.
        :type p_n: dict
        :param t:
            Target.
        :type t: dict
        :return
            Loss values of the given forecasts.
        :rtype: list
        """

        t = np.array([t_.values for t_ in t.values()])
        loss_values = []
        for p in p_n.values():
            loss_values.append(self._calculate_error(p=p.values, t=t))

        return loss_values

    def _drop_forecasts(self, loss: list) -> list:
        """
        Drops poor performing forecasts from the ensemble pool.
        :param loss:
            In-sample loss of the forecasts.
        :type loss: list
        :return
            Indexes of forecasts that should be dropped.
        :rtype: list
        """

        index_loss_dropped = []
        if self.k_best is not None:
            # Do not sort the loss_values! Otherwise, the weights do not match the given forecasts.
            if self.k_best == "autoErr":
                q75, q25 = np.percentile(loss, [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 1.5 * iqr  # Only check for outliers with high loss
                index_loss_dropped = [i for i, value in enumerate(loss) if not (value <= upper_bound)]
            elif self.k_best > len(loss):
                raise WrongParameterException(
                    "The given k is greater than the number of the given loss values.",
                    f"Make sure to define k <= {len(loss)}.",
                    self.name
                )
            else:
                index_loss_dropped = list(np.argpartition(np.array(loss), self.k_best))[self.k_best:]

        return index_loss_dropped

    @staticmethod
    def _normalize_weights(weights: list) -> list:
        """
        Normalizes the weights in the range [0,1]
        :param weights:
            Weights to be normalized.
        :type weights: list
        :return
            Normalized weights.
        :rtype: list
        """
        return [weight / sum(weights) for weight in weights]
