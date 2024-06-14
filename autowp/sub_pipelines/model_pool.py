from pywatts.core.pipeline import Pipeline

from autowp.sub_pipelines.utils import add_metrics
from pywatts.core.step_information import StepInformation


def create_power_curve_model(plant: str, modules: dict, pipeline: Pipeline, summaries: bool = False) -> StepInformation:
    """
    Create a WP curve model.
    :param plant:
        Name of the WP turbine.
    :type plant: str
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param summaries:
        Weather to add summaries or not.
    :type summaries: bool
    :return:
        Step with initialized WP curve module.
    :rtype: StepInformation
    """

    # Get steps uses as input in the following
    for step in pipeline.id_to_step.values():
        if step.name == "wind_speed_eff":
            wind_speed_eff = StepInformation(step, pipeline)
        if step.name == "power_generation":
            power_generation = StepInformation(step, pipeline)

    # Add WP curve to the pipeline
    wpc = modules["wpc"](wind_speed=wind_speed_eff)

    # Add performance metrics
    if summaries:
        add_metrics(y_hat=wpc, y=power_generation, suffix=plant)

    return wpc


def add_models(pipeline: Pipeline, modules: dict, power_curves: dict) -> Pipeline:
    """
    Adds the the WP curve models to the pipeline.
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param power_curves:
    :type power_curves:
    :return:
        Modified pyWATTS pipeline object.
    :rtype: Pipeline
    """

    # Create WP curve models for ensemble
    for plant in power_curves.keys():
        create_power_curve_model(plant=plant, pipeline=pipeline, modules=modules[plant])

    return pipeline
