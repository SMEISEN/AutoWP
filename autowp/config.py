from enum import IntEnum


class AdaptionConfiguration(IntEnum):
    """
    Enum which contains the different configurations for the online adaption of ensemble weights.
    none: No online adaption.
    AFB: Adaptive Fixed Batch (AFB), adaption with a sliding window of past values with a fixed width.
    AIB: Adaptive Increasing Batch (AIB), adaption with a sliding window of past values with an increasing width.
    """
    none = 0
    AFB = 1
    AIB = 2


class MeasurementUnit(IntEnum):
    """
    Enum which contains the measurement unit of the target WP turbine.
    kWh: kilo Watt hour (kWh)
    kW: kilo Watt (kW)
    """
    kWh = 0
    kW = 1
