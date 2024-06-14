import os
import cloudpickle
import numpy as np
import pandas as pd

from windpowerlib import data as wpl

from pywatts.core.step_information import StepInformation
from pywatts.summaries import RMSE, MAE
from pywatts_modules.ensemble import Ensemble
from pywatts_modules.estimate_wind_speed import EstimateWindSpeed
from pywatts_modules.nmae_summary import nMAE
from pywatts_modules.nrmse_summary import nRMSE
from pywatts_modules.wind_power_curve import WindPowerCurveTransformer
from pywatts_modules.ensemble import EnsembleFunction


def estimate_power_curve_sectors(power_curve: pd.DataFrame) -> ((float, float), (float, float), (float, float)):
    """
    Estimates the four sectors of the WP curve.
    :param power_curve:
        WP curve.
    :type power_curve: pd.Dataframe(index: wind speed, column: power)
    :return:
        wind_speed_activation, cut-in wind speed
        power_activation, cut-in power generation
        wind_speed_nominal, wind speed at rated power
        power_nominal, peak power rating
        wind_speed_shutdown, cut-out wind speed
        power_shutdown, cut-out power generation
    :rtype: ((float, float), (float, float), (float, float))
    """
    wind_speed_activation, power_activation = None, None
    wind_speed_nominal, power_nominal = None, None
    wind_speed_shutdown, power_shutdown = None, None

    _greater_zero = 0
    _less_previous = 0
    p_m1 = 0
    v_m1 = 0
    for v, p in power_curve.iterrows():
        # Iterate over wind speed (v) and power generation (p) values of WP curve
        p = p.values[0]
        if p > 0 and _greater_zero == 0:
            # Detect cut-in wind speed and cut-in power generation
            wind_speed_activation = v
            power_activation = p
            _greater_zero += 1  # Prevent this condition to be true afterward
        elif p == p_m1 and wind_speed_activation is not None and wind_speed_nominal is None:
            # Detect nominal wind speed and nominal power generation
            wind_speed_nominal = v_m1
            power_nominal = p_m1
        elif p < p_m1:
            # Detect cut-out wind speed and cut-out power generation
            if wind_speed_nominal is None:
                wind_speed_nominal = v_m1
                power_nominal = p_m1
            wind_speed_shutdown = v
            power_shutdown = p
            if _less_previous > 1:
                wind_speed_shutdown = None
                power_shutdown = None
            _less_previous += 1
        p_m1 = p
        v_m1 = v

    # If WP curve does not include sector 4 (cut-out)
    if wind_speed_shutdown is None and power_shutdown is None and p > 0:
        wind_speed_shutdown = v
        power_shutdown = p

    return \
        (wind_speed_activation, power_activation), \
        (wind_speed_nominal, power_nominal), \
        (wind_speed_shutdown, power_shutdown)


def load_power_curves(target_kWp: float, tolerance: float, related: bool, normalized: bool, limit: int
                      ) -> (dict, dict, float, float):
    """
    Loads WP curves from the wind power lib.
    :param target_kWp:
        Peak power rating of the target WP turbine.
    :type target_kWp: float
    :param tolerance:
        Tolerance for searching WP curves with similar peak power rating in the wind power lib.
    :type tolerance: float
    :param related:
        Weather to search for WP curves with similar peak power rating in the wind power lib.
    :type related: bool
    :param normalized:
        Weather to normalize the WP curves in the wind power lib.
    :type normalized: bool
    :param limit:
        Number of WP curves from the wind power lib to be considered in the ensemble.
    :type limit: int
    :return:
        power_curves, considered WP curves
        height_hubs, considered WP turbine hub heights
        power_activation, minimal cut-in power generation
        wind_speed_activation, minimal cut-in wind speed
    :rtype: (dict(key: WP turbine name, value: pd.Dataframe(index: wind speed, column: power)),
             dict(key: WP turbine name, value: float),
             float,
             float)
    """

    # Load WP curves from wind power lib
    try:
        data = pd.read_csv("../data/supply__wind_turbine_library.csv", encoding="latin_1")
    except FileNotFoundError as err:
        os.mkdir(err.filename)
        data = wpl.store_turbine_data_from_oedb()

    if related:
        # Search for related WP curves with similar peak power rating with tolerance
        filtered_data = data[(data["nominal_power"] >= target_kWp - target_kWp * tolerance) &
                             (data["nominal_power"] <= target_kWp + target_kWp * tolerance) &
                             (data["has_power_curve"] == True) &  # not is True!
                             (data["hub_height"].notnull())]  # not is not None! not != None
    else:
        # Select all WP curves
        filtered_data = data[(data["has_power_curve"] == True) &  # not is True!
                             (data["hub_height"].notnull())]  # not is not None! not != None

    # Create dictionaries of WP curves and hub heights
    power_curves = {}
    height_hubs = {}
    for _, entry in filtered_data.iterrows():
        power_curve_wind_speeds = np.array(eval(entry["power_curve_wind_speeds"]))
        power_curve_values = np.array(eval(entry["power_curve_values"]))
        nominal_power = max(power_curve_values)

        if normalized:
            # Normalize WP curve by its peak power rating
            power_curve_values = power_curve_values / nominal_power
            if target_kWp is not None:
                # Scale normalized WP curve with the target peak power rating
                power_curve_values = power_curve_values * target_kWp

        if len(power_curve_wind_speeds) != len(np.unique(power_curve_wind_speeds)):  # Drop duplicates
            continue

        # Create dict entries
        power_curves[f"{entry['manufacturer']} {entry['turbine_type']}"] = pd.DataFrame({
            "wind_speed": power_curve_wind_speeds, "power": power_curve_values}).set_index("wind_speed")
        height_hubs[f"{entry['manufacturer']} {entry['turbine_type']}"] = \
            [float(e.replace(',', '.')) for e in entry["hub_height"].replace("/", ";").split(";") if e]

    if limit is not None:
        # Limit the ensemble pool to <limit> WP curves

        # Create set of the indexes (wind speed) of WP curves
        common_index = set()
        for df in power_curves.values():
            common_index.update(df.index)
        common_index = sorted(common_index)

        # Resample indexes (wind speed) of WP curves
        resampled_dataframes_dict = {}
        for name, df in power_curves.items():
            resampled_df = df.reindex(common_index).interpolate(method='linear')
            resampled_df = resampled_df.fillna(0)
            resampled_df[resampled_df.index > 25] = 0
            resampled_dataframes_dict[name] = resampled_df

        # Create dataframe consisting of all resampled WP curves
        combined_df = pd.concat(resampled_dataframes_dict.values(), axis=1, keys=resampled_dataframes_dict.keys())

        # Sort WP curves by their area under the curve (equivalent to the sum since all are resampled to the same index)
        column_sums = combined_df.sum()
        combined_df_sorted = combined_df[column_sums.sort_values(ascending=False).index]

        # Determine number of WP curves to be removed
        num_columns_to_remove = len(combined_df_sorted.columns) - limit

        # Determine WP curves to be kept in equidistant step size between the first and the last WP curve
        step_size = max(num_columns_to_remove / limit + 2, 1)
        columns_to_keep = [0] + [int(i * step_size) for i in range(1, limit - 1)] + [-1]
        reduced_df = combined_df_sorted.iloc[:, columns_to_keep]

        # Create dict entries
        power_curves = {k: v for k, v in power_curves.items() if k in reduced_df.columns}
        height_hubs = {k: v for k, v in height_hubs.items() if k in reduced_df.columns}

    # Create list of cut-in wind speeds and cut-in power generations of the WP curves of the ensemble pool
    wind_speed_activation_list = []
    power_activation_list = []
    for power_curve in power_curves.values():
        (wind_speed_activation, power_activation), _, _ = estimate_power_curve_sectors(power_curve)
        wind_speed_activation_list.append(wind_speed_activation)
        power_activation_list.append(power_activation)

    # Determine minimal cut-in wind speed and minimal cut-in power generation in the WP curves of the ensemble pool
    wind_speed_activation = min(i for i in wind_speed_activation_list if i > 0)
    power_activation = min(i for i in power_activation_list if i > 0)

    return power_curves, height_hubs, power_activation, wind_speed_activation


def create_modules(power_curves: dict, target_kWp: float) -> dict:
    """
    Creates the modules for the pyWATTS pipeline.
    :param power_curves:
        WP curves for the ensemble optimization
    :type power_curves: dict(key: WP turbine name, value: pd.Dataframe(index: wind speed, column: power))
    :param target_kWp:
        Peak power rating of the target WP turbine.
    :type target_kWp: float
    :return:
        Initialized pyWATTS modules.
    :rtype: dict
    """

    modules = {
        "ensemble": Ensemble(
            weights="autoOpt",
            # Determine ensemble weights by least-squares optimization
            ensemble_fun=EnsembleFunction.WeightedSum if target_kWp is None else EnsembleFunction.WeightedAverage,
            # Define objective function for the ensemble optimization
            name=f"y_hat"),
        "wind_speed_eff": EstimateWindSpeed(name="wind_speed_eff", alpha=1 / 7)
        # Recommended values for onshore alpha=1/7 and offshore alpha=1/9
    }

    for plant, power_curve in power_curves.items():
        (wind_speed_activation, _), \
            (wind_speed_nominal, _), \
            (_, _) = estimate_power_curve_sectors(power_curve)

        modules.update({plant: {
            "wpc":
                WindPowerCurveTransformer(name=f"WPC_{plant}", power_curve=power_curve, kWp=target_kWp,
                                          wind_speed_activation=wind_speed_activation,
                                          wind_speed_nominal=wind_speed_nominal,
                                          wind_speed_shutdown=25.0)
        }})

    return modules


def save_modules(pipeline_modules: dict, dry: str) -> None:
    """
    Save pyWATTS pipeline as pickle file.
    :param pipeline_modules:
        Initialized and fitted pyWATTS modules.
    :type pipeline_modules: dict
    :param dry:
        Directory of saved AutoWP model.
    :type dry: str
    :return: None
    :rtype: None
    """
    for plant, modules in pipeline_modules.items():
        path = f"{dry}/{plant}"
        if not os.path.exists(path):
            os.makedirs(path)
        for module_name, module_object in modules.items():
            cloudpickle.dump(module_object, open(f"{path}/{module_name}.pickle", 'wb'))


def load_modules(pipeline_modules: dict, dry: str) -> dict:
    """
    Save pyWATTS pipeline from a pickle file.
    :param pipeline_modules:
        Initialized and fitted pyWATTS modules.
    :type pipeline_modules: dict
    :param dry:
        Directory of saved AutoWP model.
    :type dry: str
    :return:
        Initialized and fitted pyWATTS modules.
    :rtype: dict
    """
    for plant, modules in pipeline_modules.items():
        for module_name, module_object in modules.items():
            with open(f"{dry}/{plant}/{module_name}.pickle", 'rb') as file:
                pipeline_modules[plant][module_name] = cloudpickle.load(file)
    return pipeline_modules


def add_metrics(y_hat: StepInformation, y: StepInformation, suffix: str) -> None:
    """
    Add metrics to the pyWATTS pipeline.
    :param y_hat:
        Prediction.
    :type y_hat: StepInformation
    :param y:
        Target.
    :type y: StepInformation
    :param suffix:
        Individual suffix for the metric name.
    :type suffix: str
    :return: None
    :rtype: None
    """

    RMSE(name=f"RMSE_{suffix}")(y_hat=y_hat, y=y)
    MAE(name=f"MAE_{suffix}")(y_hat=y_hat, y=y)

    nRMSE(name=f"nRMSEavg_{suffix}")(y_hat=y_hat, y=y)
    nMAE(name=f"nMAEavg_{suffix}")(y_hat=y_hat, y=y)
