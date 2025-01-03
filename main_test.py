import dstools as ds
import json
import numpy as np
import pandas as pd

PATH = "datasets/dataset_train.csv"

def createDatasTest() -> pd.DataFrame:
    """
    Function that create a DataFrame for test
    Return : a pd.DataFrame object
    """
    # Générer des valeurs aléatoires pour x1 et x2
    x1 = np.random.randint(0, 10, size=300)
    x2 = np.random.randint(20, 32, size=300)

    # Calculer y selon l'équation
    y_linear = - 5.2 + 0.6 * x1 + 0.2 * x2
    y = ds.sigmoid(y_linear)

    # Créer un DataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    dfWithIntercept = pd.concat([pd.Series([1] * len(df), name='intercept'), df], axis=1)
    return dfWithIntercept

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
            column_name = f'x{index}'
            if column_name in dfNormalized.columns:
                column = dfNormalized[column_name]
                sumErrorCoeff = np.dot(error, column)
                newWeights[index] = value - alpha / nbEl * sumErrorCoeff
            else :
                raise ValueError(f"Column {column_name} not found in the DataFrame")
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
    

def testMain():
    try :
        # Création de la DataFrame pour test
        df = createDatasTest()

        datas = {
            'β0' : [0, 0],
            'β1' : [df['x1'].mean(), df['x1'].std()],
            'β2' : [df['x2'].mean(), df['x2'].std()]     
        }
        normalizationParameters = pd.DataFrame(datas, index=['mean', 'std'])
        normalizedBeta1 = ds.normalizePdSeries(df['x1'], normalizationParameters['β1'])
        normalizedBeta2 = ds.normalizePdSeries(df['x2'], normalizationParameters['β2'])
        dfNormalized = pd.concat([df['intercept'], normalizedBeta1, normalizedBeta2], axis=1)
        alpha = 0.1

        weights = gradiantDescent(dfNormalized, alpha, df['y'])
        
        print(weights)
        estimatedResults = dfNormalized.apply(lambda x: ds.predictionH0(weights, x), axis=1)
        compResults = pd.concat([df['y'], estimatedResults], axis=1)
        print(compResults)
        saveDatas(weights, normalizationParameters)

        # print(dfNormalized)


    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    testMain()