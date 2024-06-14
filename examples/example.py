from autowp.autowp import AutoWP
from autowp.config import MeasurementUnit
from data.utils import load_example_data

if __name__ == "__main__":
    # Please note that the example data includes weather forecasting data only.
    # If the measurement data of the wind speed at hub height, the effective wind speed can be estimated.
    # Summaries will be created in the result-folder.
    _, train, test = load_example_data()

    autowp = AutoWP(target_power_generation="TARGETVAR", measurement_unit=MeasurementUnit.kW)

    # Estimate the ensemble weights offline using the training data.
    result_fit = autowp.fit(data=train)

    # Predict with the offline-fitted AutoWP model.
    result_predict = autowp.predict(data=test)

    # Get AutoWP's ensemble weights.
    weights = autowp.weights_
