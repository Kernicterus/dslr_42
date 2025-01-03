import pandas as pd
import numpy as np

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


def extractAndPrepareNumericalDatas(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    """
    Function that extract numerical datas, filled missing values with median and normalize datas
    Parameters : a pd.DataFrame object
    Return : a new dataFrame containing only numerical datas and a dataFrame containing 
    mean and std parameters for each variable
    """
    numericalDf = df.select_dtypes(include=['int64', 'float64'])
    numericalDf = numericalDf.drop(columns=['Index'])
    parameters = pd.DataFrame(columns=numericalDf.columns, index=['mean', 'std', 'median'])
    for column in numericalDf.columns:
        # CHANGER AVES NOS PROPRES FONCTIONS
        median = numericalDf[column].median()
        mean = numericalDf[column].mean()
        std = numericalDf[column].std()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        numericalDf[column] = numericalDf[column].fillna(median)
        parameters[column] = [mean, std, median]
    for column in numericalDf.columns:
        numericalDf[column] = normalizePdSeries(numericalDf[column], parameters[column])
    return numericalDf, parameters


def sigmoid(z):
    """
    Function that calculates the sigmoid of a given value
    alias g(z) and returns it
    """    
    return 1 / (1 + np.exp(-z))


def predictionH0(weights : pd.Series, dfLine : pd.Series):
    """
    Function that calculates h0(x)
    by sending 0Tx to the sigmoid function
    which is the dot product of the weights and the datas
    requirements : 
        - df must only contain normalized numerical datas useful for the prediction
            AND a column of '1' must be added at index 0 for the interception
        - weights must contain the weights calculated for each variable 
            + the interception at index 0 
    """

    if len(weights) != len(dfLine) :
        raise ValueError("The number of weights must be equal to the number")
    if dfLine[0] != 1:
        raise ValueError("The first column of the datas must be '1' for product with interception")
    thetaTx = np.dot(weights, dfLine)
    return sigmoid(thetaTx)

def extractAndPrepareDiscreteDatas(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that extract discrete datas and numerize them
    Parameters : a pd.DataFrame object
    Return : a pd.DataFrame object containing the numerized datas
    """
    discreteDatas = df.select_dtypes(include=['object'])
    discreteDatas[['year', 'month', 'day']] = discreteDatas['Birthday'].str.split('-', expand=True)
    discreteDatas = discreteDatas.drop(columns=['Hogwarts House'])
    discreteDatas = discreteDatas.drop(columns=['Birthday'])

    parameters = pd.DataFrame(columns=discreteDatas.columns, index=['mean', 'std', 'median'])
    discreteDatas = discreteDatas.apply(lambda x: x.astype('category').cat.codes)
    for column in discreteDatas.columns:
        # CHANGER AVES NOS PROPRES FONCTIONS
        median = discreteDatas[column].median()
        mean = discreteDatas[column].mean()
        std = discreteDatas[column].std()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        discreteDatas[column] = discreteDatas[column].fillna(mean)
        parameters[column] = [mean, std, median]
    for column in discreteDatas.columns:
        discreteDatas[column] = normalizePdSeries(discreteDatas[column], parameters[column])
    return discreteDatas, parameters
