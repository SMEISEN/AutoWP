import logging

import numpy as np

from pywatts.summaries.metric_base import MetricBase

logger = logging.getLogger(__name__)


class nRMSE(MetricBase):
    """
    Module to calculate the normalized Root Mean Squared Error (RMSE), normalization with the ground truth's mean
    """

    def _apply_metric(self, p, t) -> float:
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
        return np.sqrt(np.mean((p - t) ** 2)) / (np.mean(t) + 10e-8)
