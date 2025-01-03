import dstools as ds
import json
import numpy as np
import pandas as pd
import sys
import os

ITERATION = 100

def checkArgs(args : list) -> bool:
    if len(args) != 2:
        print("Error: wrong number of arguments")
        return False
    if args[1].split('.')[-1] != "csv":
        print("Error: wrong file extension")
        return False
    if not os.path.exists(args[1]):
        print("Error: file not found")
        return False
    if not open(args[1], 'r').readable():
        print("Error: file not readable")
        return False
    if open(args[1], 'r').read() == "":
        print("Error: empty file")
        return False
    return True


def saveDatas(weights : pd.Series, numDatasParams : pd.DataFrame, discreteDatasParams : pd.DataFrame):
    """
    Save weights into a file
    """
    json_structure = {"data": weights.to_dict()}
    with open("training.json", "w") as file:
        json.dump(json_structure, file, indent=4)

    json_structure = {"data": numDatasParams.to_dict()} 
    with open("parametersNumDatas.json", "w") as file:  
        json.dump(json_structure, file, indent=4)

    json_structure = {"data": discreteDatasParams.to_dict()} 
    with open("parametersDiscrDatas.json", "w") as file:  
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


def gradiantDescent(dfNormalized : pd.DataFrame, alpha: float, results: pd.Series, iteration : int) -> pd.Series:
    """
    Function that calculates the gradiant descent
    Parameters : 
        - a pd.DataFrame containing the normalized datas
        - a float containing the learning rate
    Return : a pd.Series containing the weights calculated
    """
    weights = pd.Series([0.0] * len(dfNormalized.columns))
    for iteration in range(iteration):
        weights = updWeights(weights, dfNormalized, alpha, results, len(dfNormalized))
    return weights

def prepareResults(df : pd.DataFrame) -> pd.DataFrame:
    """
    Function that prepares the results for each classifier
    1 for the house, 0 for the others
    Parameters : a pd.DataFrame object
    Return : a pd.DataFrame object containing the results for each classifier (1 or 0)
    """
    gryff = df['Hogwarts House'].apply(lambda x: 1 if x == 'Gryffindor' else 0)
    slyth = df['Hogwarts House'].apply(lambda x: 1 if x == 'Slytherin' else 0)
    raven = df['Hogwarts House'].apply(lambda x: 1 if x == 'Ravenclaw' else 0)
    huffl = df['Hogwarts House'].apply(lambda x: 1 if x == 'Hufflepuff' else 0)
    gryff.rename('Gryffindor', inplace=True)
    slyth.rename('Slytherin', inplace=True)
    raven.rename('Ravenclaw', inplace=True)
    huffl.rename('Hufflepuff', inplace=True)
    results = pd.concat([gryff, slyth, raven, huffl], axis=1)
    return results


def main():
    try :
        if checkArgs(sys.argv) == False:
            return 1
        
        # step 1 : load the dataset
        df = pd.read_csv(sys.argv[1])

        # step 2 : drop the rows with missing values in the Hogwarts House column
        df = df.dropna(subset=['Hogwarts House'])

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = ds.extractAndPrepareNumericalDatas(df)

        # step 4 : extraction, numerization, filling missing values (MEAN) and standardization of discrete datas
        discreteDatas, discreteDatasParams = ds.extractAndPrepareDiscreteDatas(df)

        # step 5 : regroup the datas and add the intercept
        dfWithIntercept = pd.concat([pd.Series([1] * len(df), name='intercept'), normalizedDatas, discreteDatas], axis=1)
        
        # step 6: rename the columns of the dataframe with numerical indexes
        dfWithIntercept.columns = range(dfWithIntercept.shape[1])

        # step 7 : prepare the results for each classifier (0 or 1) : one vs all technique
        results = prepareResults(df)

        # step 8 : calculate the weights for each classifier
        weightsGryff = gradiantDescent(dfWithIntercept, 0.1, results['Gryffindor'], ITERATION)
        weightsSlyth = gradiantDescent(dfWithIntercept, 0.1, results['Slytherin'], ITERATION)
        weightsRaven = gradiantDescent(dfWithIntercept, 0.1, results['Ravenclaw'], ITERATION)
        weightsHuffl = gradiantDescent(dfWithIntercept, 0.1, results['Hufflepuff'], ITERATION)
        weights = pd.concat([weightsGryff, weightsSlyth, weightsRaven, weightsHuffl], axis=1)
        weights.columns = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

        # step 9 : save the weights and the parameters
        saveDatas(weights, numDatasParams, discreteDatasParams)

        # TESTING -----------------------------------------------------
        probabilityGryff = dfWithIntercept.apply(lambda x: ds.predictionH0(weights['Gryffindor'], x), axis=1)
        probabilitySlyth = dfWithIntercept.apply(lambda x: ds.predictionH0(weights['Slytherin'], x), axis=1)
        probabilityRaven = dfWithIntercept.apply(lambda x: ds.predictionH0(weights['Ravenclaw'], x), axis=1)
        probabilityHuffl = dfWithIntercept.apply(lambda x: ds.predictionH0(weights['Hufflepuff'], x), axis=1)
        dfResults = pd.concat([probabilityGryff, probabilitySlyth, probabilityRaven, probabilityHuffl], axis=1)
        estimatedResults = dfResults.idxmax(axis=1)

        house_mapping = {
            0: 'Gryffindor',
            1: 'Slytherin',
            2: 'Ravenclaw',
            3: 'Hufflepuff'
        }        
        estimatedResults = estimatedResults.map(house_mapping)
        trueResults = (estimatedResults == df['Hogwarts House']).astype(int)
        print("True results :")
        print(trueResults)
        precision = trueResults.sum() / len(trueResults)
        print(f"Precision : {precision}")
        # TESTING -----------------------------------------------------

    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()