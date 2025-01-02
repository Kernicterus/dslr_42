import pandas as pd

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


def mean(column: pd.Series) -> float :
    """
    Function that calculate the mean of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated mean
    """
    pass


def std(column: pd.Series) -> float :
    """
    Function that calculate the std of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated std
    """
    pass


def percentile(column: pd.Series, percentile: int) -> int : 
    """
    Function that look for the percentile asked for a variable
    Parameters : 
        - a pd.Series column containing datas
        - the percentile asked for that column
    Return : index of the percentile asked
    """
    pass


def normalizePdSeries(variable : pd.Series, parameters : pd.Series) -> pd.Series :
    """
    Function to standardize a given variable from its different values
    Parameters : a pd.Series object containing the mean and std of the variable
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