import pandas as pd
import math

def load_csv(path: str) -> pd.DataFrame:
    """
    Function to load a csv
    Parameters : path of the csv
    Return : pd.DataFrame containing the csv datass
    """
    try:
        csv = pd.read_csv(path)
    except Exception as e:
        print(f"loading csv error : {e}")
        return None
    return csv

def min(column: pd.Series):
    min = column[0]
    for value in column:
        if not pd.isna(value):
            if value < min:
                min = value
    return min

def max(column: pd.Series):
    max = column[0]
    for value in column:
        if not pd.isna(value):
            if value > max:
                max = value
    return max

def mean(column: pd.Series) -> float :
    """
    Function that calculate the mean of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated mean
    """
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += value
            size += 1
    return sum / size

def std(column: pd.Series) -> float :
    """
    Function that calculate the std of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated std
    """
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += (mean(column) - value) ** 2
            size += 1
    return math.sqrt(sum / size)


def percentile(column: pd.Series, percentile: int) -> int : 
    """
    Function that look for the percentile asked for a variable
    Parameters : 
        - a pd.Series column containing datas
        - the percentile asked for that column
    Return : index of the percentile asked
    """
    try:
        cleanData = column.dropna()
        sortedData = cleanData.sort_values()
        n = len(sortedData)
        index = (n - 1) * percentile
        if index.is_integer():
            return sortedData.iloc[int(index)]
        lowerIndex = int(index)
        upperIndex = lowerIndex + 1

        lowerValue = sortedData.iloc[lowerIndex]
        upperValue = sortedData.iloc[upperIndex]

        fraction = index - lowerIndex
        return lowerValue + fraction * (upperValue - lowerValue)

    except Exception as e:
        print(f"Error: {e}")


def normalizePdSeries(variable : pd.Series, parameters : pd.Series) -> pd.Series :
    """
    Function to calculate the mean and std of a given variable from its different values
    Parameters : a pd.Series object
    Return : a new pd.Series containing the normalized values of the variable
    """ 
    variableNormalized = (variable - parameters['mean']) / parameters['std']
    return variableNormalized


def extractNormalizedNumericalDatas(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    """
    Function that extract numerical datas, normalized it and return the normalized datas
    Parameters : a pd.DataFrame object
    Return : a new dataFrame containing only numerical datas and a dataFrame containing 
    mean and std parameters for later denormalization
    """
    numericalDf = df.select_dtypes(include=['int64', 'float64'])

    # ... implement code to normalize each column of numerical datas  ...
    # normalizedDatas = 
    # parameters =
    # normalizationParameters = pd.DataFrame(parameters, index=['mean', 'std'])
    # 
    # return normalizedDatas, normalizationParameters