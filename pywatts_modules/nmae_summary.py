import logging

import numpy as np

from pywatts.summaries.metric_base import MetricBase

logger = logging.getLogger(__name__)


class nMAE(MetricBase):
    """
    Module to calculate the normalized Mean Absolute Error (nMAE), normalization with the ground truth's mean
    """

    def _apply_metric(self, p: np.array, t: np.array) -> float:
        """
        :param p:
            Prediction values.
        :type p: np.array
        :param t:
            Target values.
        :type t: np.array
        :return:
            Calculated nMAE.
        :rtype: float
        """
        return np.mean(np.abs((p - t))) / (np.mean(t) + 10e-8)
