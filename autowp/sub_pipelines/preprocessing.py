from pywatts.core.pipeline import Pipeline
from pywatts.modules import CustomScaler, FunctionModule

from autowp.config import MeasurementUnit


def add_preprocessing(pipeline: Pipeline, modules: dict, target_power_generation: str, target_wind_speed_hub: str,
                      measurement_unit: MeasurementUnit, samples_per_hour: int) -> Pipeline:
    """
    Adds pre-processing steps to the pipeline.
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param target_power_generation:
        Column name of the target WP turbine in the dataframe given in train and predict.
    :type target_power_generation: str
    :param target_wind_speed_hub:
        Column name of the wind speed measurement at hub height of the target WP turbine in the dataframe given in train
        and predict (optional, default=None).
        If not provided, it is recommended to let AutoWP estimate the peak power of the WP turbine to implicitly consider
        the height correction in the ensemble optimization.
    :type target_wind_speed_hub: str
    :param measurement_unit:
        Unit of the WP generation, can be kW or kWp.
    :type measurement_unit: MeasurementUnit
    :param samples_per_hour:
        Resolution of the time series in samples per hour (optional, default=None).
        Required for transforming an energy metering time series into the mean power time series.
    :type samples_per_hour: int
    :return:
        Modified pyWATTS pipeline object.
    :rtype: Pipeline
    """

    # Transform power generation data to kW
    if measurement_unit == MeasurementUnit.kW:
        FunctionModule(lambda x: x, name="power_generation")(x=pipeline[target_power_generation])
    elif measurement_unit == MeasurementUnit.kWh:
        # samples_per_hour: convert the energy metering time series into the mean power time series
        CustomScaler(multiplier=samples_per_hour, name="power_generation")(x=pipeline[target_power_generation])

    # Calculate total effective wind speed
    if target_wind_speed_hub is not None:
        modules["wind_speed_eff"](U100=pipeline["U100"], V100=pipeline["V100"], target=pipeline[target_wind_speed_hub])
    else:
        FunctionModule(lambda u, v: (u ** 2 + v ** 2) ** (1 / 2), name="wind_speed_eff")(u=pipeline["U100"],
                                                                                         v=pipeline["V100"])

    return pipeline
