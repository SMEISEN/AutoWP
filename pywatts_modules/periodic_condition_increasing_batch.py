import pandas as pd

from pywatts.core.base_condition import BaseCondition


class PeriodicConditionIncreasingBatch(BaseCondition):
    """
    This condition is raised after each num_steps and uses an increasing batch for refitting the corresponding modules.
    """

    def __init__(self, num_steps=10, refit_batch: pd.Timedelta = pd.Timedelta(hours=0),
                 refit_batch_append: pd.Timedelta = pd.Timedelta(hours=24),
                 refit_params: dict = None,
                 name="PeriodicCondition"):
        """
        Initialize the condition.
        :param num_steps:
            After num_steps the periodicCondition should be True.
        :type num_steps: int
        :param refit_batch:
            Data batch used to refit the corresponding modules (optional, default=pd.Timedelta(hours=0))
        :type refit_batch: pd.Timedelta
        :param refit_batch_append:
            Data batch used to append the refit batch (optional, default=pd.Timedelta(hours=24))
        :type refit_batch_append: pd.Timedelta
        :param refit_params:
            Params to be set for the corresponding module before refitting.
        :type refit_params: dict
        :param name:
            Step name in the pyWATTS pipeline (optional, default='Ensemble')
        :type name: str
        """

        super().__init__(name=name, refit_batch=refit_batch, refit_params=refit_params)
        self.num_steps = num_steps
        self.counter = 0
        self.refit_batch_append = refit_batch_append

    def evaluate(self, start, end) -> bool:
        """
        Returns True if it is num_steps times called else False.
        :param start:
            Start of the batch.
        :type start:pd.Timestamp
        :param end:
            End of the batch.
        :type end: pd.Timestamp
        :return:
            Weather to refit or not.
        :rtype: bool
        """

        increase_batch = False
        if not self._is_evaluated(end):
            self.counter += 1
            increase_batch = True
        self.counter = self.counter % self.num_steps

        if self.counter == 0:
            if increase_batch:
                self.refit_batch = self.refit_batch + self.refit_batch_append
            print(f"{self.name}: refit with refit batch {self.refit_batch}")
            return True
        else:
            return False
