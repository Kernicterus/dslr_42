import dstools as ds
import json
import numpy as np
import pandas as pd

PATH = "datasets/dataset_train.csv"

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
        - the parameters need for denormalization of each variable
    Return : a pd.Series containing the real values of the calculated weights
    """
    for index, value in weights.iteritems():
    # interception
        if index == 0 :
            pass
    # coefficients
        else :
            weights[index] = weights[index] / normalizationParameters.loc['std', index + 1]


    pass


def main():
    try :
        csv = ds.load_csv(PATH)
        #  ----- TEST -----
        datas = {
            'Astronomy' : [csv['Astronomy'].mean(),csv['Astronomy'].std()],
            'Herbology' : [csv['Herbology'].mean(),csv['Herbology'].std()],
        }
        normalizationParameters = pd.DataFrame(datas, index=['mean', 'std'])
        print (normalizationParameters)
        print(csv['Astronomy'])
        csv['Astronomy'] = ds.normalizePdSeries(csv['Astronomy'], normalizationParameters['Astronomy'])
        csv['Herbology'] = ds.normalizePdSeries(csv['Herbology'], normalizationParameters['Herbology'])

        #  -----
        
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()