import pandas as pd

from pywatts.conditions import PeriodicCondition
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.step_information import StepInformation
from pywatts.modules import FunctionModule
from pywatts_modules.periodic_condition_increasing_batch import PeriodicConditionIncreasingBatch
from autowp.config import AdaptionConfiguration

from autowp.sub_pipelines.utils import add_metrics


def add_ensemble(pipeline: Pipeline, modules: dict, target_power_generation: str,
                 adaption_config: AdaptionConfiguration, K: pd.Timedelta) -> Pipeline:
    """
    Adds the ensemble to the pipeline.
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param K:
        Adaption batch (past horizon) for the online adaption of ensemble weights (optional, default=None).
    :type K: pd.Timedelta
    :param target_power_generation:
        Column name of the target WP turbine in the dataframe given in train and predict.
    :type target_power_generation: str
    :param adaption_config:
        Configuration for the online adaption of ensemble weights (optional, default=AdaptionConfiguration.None).
    :type adaption_config: AdaptionConfiguration
    :return:
        Modified pyWATTS pipeline object.
    :rtype: Pipeline
    """

    # Get steps uses as input in the following
    for step in pipeline.id_to_step.values():
        if step.name == "wind_speed_eff":
            wind_speed = StepInformation(step, pipeline)
        if step.name == "power_generation":
            power_generation = StepInformation(step, pipeline)

    # Define computation mode
    if adaption_config == AdaptionConfiguration.none:
        computation_mode = ComputationMode.Default
    else:
        computation_mode = ComputationMode.Refit

    # Create ensemble model pool
    ensemble_model_pool = {}
    for plant in modules.keys():
        if plant == "ensemble":
            continue
        for step in pipeline.id_to_step.values():
            if step.name == f"WPC_{plant}":
                ensemble_model_pool[plant] = StepInformation(step, pipeline)

    # Define refit conditions
    if adaption_config == adaption_config.AFB:
        refit_conditions = [PeriodicCondition(name="PeriodicEnsemble",
                                              num_steps=1,  # refits every C period
                                              refit_batch=K)]
    elif adaption_config == adaption_config.AIB:
        refit_conditions = [PeriodicConditionIncreasingBatch(name="PeriodicIncreasingEnsemble",
                                                             num_steps=1,  # refits every C period
                                                             refit_batch=pd.Timedelta(days=0),  # initial refit batch
                                                             refit_batch_append=K)]  # increases refit batch every C by K
    else:
        refit_conditions = []

    # Add ensemble to the pipeline
    ensemble = modules["ensemble"](
        **ensemble_model_pool, target=power_generation,
        refit_conditions=refit_conditions,
        computation_mode=computation_mode,
        config_summary=["weights_"])

    # The ensemble defaults to equal weights and is then periodically fitted
    ensemble.step.module.is_fitted = True

    # Make outputs of the individual WP turbine and the wind speed available in the results
    for step in ensemble_model_pool.values():
        step.step.last = True
    wind_speed.step.last = True

    # Add performance metrics
    add_metrics(y_hat=ensemble, y=power_generation, suffix=f"_ensemble_{target_power_generation}")

    # Rename ground truth power_generation -> "y"
    FunctionModule(lambda x: x, name="y")(x=power_generation)

    return pipeline
