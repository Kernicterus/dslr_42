import dstools as ds
import json
import numpy as np
import pandas as pd

PATH = "datasets/dataset_train.csv"

def saveDatas(weightsN : pd.Series, parameters : pd.DataFrame):
    """
    Save weights into a file
    """
    json_structure = {"data": weightsN.to_dict()}
    with open("training.json", "w") as file:
        json.dump(json_structure, file, indent=4)

    json_structure = {"data": parameters.to_dict()} 
    with open("parameters.json", "w") as file:  
        json.dump(json_structure, file, indent=4)


def updWeights(weights : pd.Series, dfNormalized : pd.DataFrame, alpha: float, results: pd.Series, nbEl : int) -> pd.Series:
    """
    Function that updates the weights
    Parameters : 
        - a pd.Series containing the weights
        - a pd.DataFrame containing the normalized datas
        - a float containing the learning rate
    Return : a pd.Series containing the updated weights
    """
    newWeights = weights.copy()
    estimatedResults = pd.Series([0] * len(results))
    estimatedResults = dfNormalized.apply(lambda x: ds.predictionH0(weights, x), axis=1)
    error = estimatedResults - results
    sumErrorBias = error.sum()
    for index, value in weights.items():
        if index == 0:
            newWeights[index] = value - alpha / nbEl * sumErrorBias
        else :
            column = dfNormalized[index]
            if (len(column) != len(error)):
                raise ValueError(f"Column {index} has not the same length as the error")
            sumErrorCoeff = np.dot(error, column)
            newWeights[index] = value - alpha / nbEl * sumErrorCoeff
    return newWeights


def gradiantDescent(dfNormalized : pd.DataFrame, alpha: float, results: pd.Series) -> pd.Series:
    """
    Function that calculates the gradiant descent
    Parameters : 
        - a pd.DataFrame containing the normalized datas
        - a float containing the learning rate
    Return : a pd.Series containing the weights calculated
    """
    weights = pd.Series([0.0] * len(dfNormalized.columns))
    for iteration in range(1000):
        weights = updWeights(weights, dfNormalized, alpha, results, len(dfNormalized))
    return weights


def main():
    try :
        # step 1 : load the dataset
        df = pd.read_csv(PATH)
        # step 2 : extract numerical datas and normalize it
        normalizedDatas, parameters = ds.extractNormalizedNumericalDatas(df)
        # step 3 : binary classification of discrete datas
        # step 4 : regroup the datas and add the intercept
        # step 4b: rename the columns of the dataframe with numerical indexes
        # step 5 : prepare the results for each classifier (0 or 1) : one vs all technique
        # step 6 : calculate the weights
        # step 7 : save the weights and the parameters
        
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()