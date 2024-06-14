import pandas as pd


def split_data(data: pd.DataFrame, test_percent: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split the (not shuffled) data into a training and a test data sub-set with a fixed
    :param data:
        Data to be splitted.
    :type data: pd.DataFrame
    :param test_percent:
        Percentage of samples considered for the test data sub-set.
    :type test_percent: float
    :return:
        data, the complete data set
        train, the training data sub-set
        test, the test data sub-set
    :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    split_index = int(len(data) - (len(data) * test_percent))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    return data, train, test


def load_example_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Load the example data.
    :return:
        data, the complete data set
        train, the training data sub-set
        test, the test data sub-set
    :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    data = pd.read_csv("../data/wind.csv")
    data = data.dropna()
    data.index = pd.date_range(start="1/1/2012 01:00:00", freq="1h", periods=len(data))
    data.index.name = 'time'
    data = data.drop(columns=["Unnamed: 0"])

    return split_data(data=data, test_percent=0.2)
