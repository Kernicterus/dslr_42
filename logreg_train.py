import dstools as ds
import json
import numpy as np
import pandas as pd

PATH = "datasets/dataset_train.csv"

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predictionH0(i : int, weights : pd.Series, df : pd.DataFrame):
    thetaTx = dotproduct(weights, df.loc[i])
    return sigmoid(thetaTx)


def createDatasTest() -> pd.DataFrame:
    """
    Function that create a DataFrame for test
    Return : a pd.DataFrame object
    """
    # Générer des valeurs aléatoires pour x1 et x2
    x1 = np.random.randint(0, 10, size=40)
    x2 = np.random.randint(20, 32, size=40)

    # Calculer y selon l'équation
    y_linear = - 5.2 + 0.6 * x1 + 0.2 * x2
    y = sigmoid(y_linear)

    # Créer un DataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    dfWithIntercept = pd.concat([pd.Series([1] * len(df), name='intercept'), df], axis=1)
    return dfWithIntercept

def saveDatas(weightsDN : pd.Series):
    """
    Save weights into a file
    """
    json_structure = {"data": weightsDN.to_dict()}
    with open("training.json", "w") as file:
        json.dump(json_structure, file, indent=4)


def denormalize(weights : pd.Series, normalizationParameters: pd.DataFrame) -> pd.Series :
    """
    Function that denormalize the given datas
    Parameters : 
        - a pd.Series containing the normalized weights 
                calculated for each variable
        - the parameters needed for denormalization of each variable
    Return : a pd.Series containing the real values of the calculated weights
    """
    denormalizedWeights = weights.copy()
    for index, value in weights.items():
        # interception
        if index == 'β0' :
            print("***")
            adjustment = np.sum(
                weights[1:] * normalizationParameters.loc['mean', weights.index[1:]].values /
                normalizationParameters.loc['std', weights.index[1:]].values
            )
            denormalizedWeights['β0'] = weights['β0'] - adjustment
            print(denormalizedWeights['β0'])
            print("***")
        else:
            print(f"index {index} normalized : {weights[index]}")
            denormalizedWeights[index] = weights[index] * normalizationParameters.loc['std', index]
            - normalizationParameters.loc['mean', index]
            print(f"index {index} denormalized : {denormalizedWeights[index]}")


    # for index in weights.items():
    # # interception
    #     if index == 0 :
    #         value = value - np.sum(weights[1:] * normalizationParameters.loc['mean', 1:] / normalizationParameters.loc['std', 1:])
    # # coefficients
    #     else :
    #         weights[index] = weights[index] / normalizationParameters.loc['std', index - 1]
    return denormalizedWeights

def gradiantDescent(normalizedBeta1, normalizedBeta2, alpha):
    pass

def testMain():
    try :
        # Création de la DataFrame pour test
        df = createDatasTest()
        print(df)
        datas = {
            'β0' : [0, 0],
            'β1' : [df['x1'].mean(), df['x1'].std()],
            'β2' : [df['x2'].mean(), df['x2'].std()]     
        }
        normalizationParameters = pd.DataFrame(datas, index=['mean', 'std'])
        print(normalizationParameters)
        normalizedBeta1 = ds.normalizePdSeries(df['x1'], normalizationParameters['β1'])
        normalizedBeta2 = ds.normalizePdSeries(df['x2'], normalizationParameters['β2'])
        print(normalizedBeta1)


    except Exception as e:
        print(f"Error: {e}")

def main():
    try :
        csv = ds.load_csv(PATH)
        #  ----- TEST  -----
        datas = {
            'β0' : [0, 0],
            'β1' : [3.0, 2.0],
            'β2' : [4.0, 1.5],
        }
        weights = [1.5, 0.8, 1.2]  # β0, β1, β2

        # Création de la Series des coefficients
        weights_series = pd.Series(weights, index=['β0', 'β1', 'β2']) 

        # Création de la DataFrame nornalizationParameters      
        normalizationParameters = pd.DataFrame(datas, index=['mean', 'std'])

        weightsDN = denormalize(weights_series, normalizationParameters)
        saveDatas(weightsDN)


        #  -----
        
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    # main()
    testMain()