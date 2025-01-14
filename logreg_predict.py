import dstools as ds
import pandas as pd
import json

HOUSE = [
        'Gryffindor',
        'Slytherin',
        'Ravenclaw',
        'Hufflepuff',
        ]

def normalizePdSeries(variable : pd.Series, mean, std) -> pd.Series :
    """
    Function to standardize a given variable from its different values
    Parameters : a pd.Series object containing the mean and std of the variable
    Return : a new pd.Series containing the normalized values of the variable
    """ 
    variableNormalized = (variable - mean) / std
    return variableNormalized

def extractAndPrepareNumericalDatas(df : pd.DataFrame, trainingParam : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    """
    Function that extract numerical datas, filled missing values with median and normalize datas
    Parameters : a pd.DataFrame object
    Return : a new dataFrame containing only numerical datas and a dataFrame containing 
    mean and std parameters for each variable
    """
    for column in df.columns:
        df[column] = df[column].fillna(trainingParam[column]['median'])
    for column in df.columns:
        if (column == 'Best Hand'):
            df[column] = df[column].astype('category').cat.codes
        mean = trainingParam[column]['mean']
        std = trainingParam[column]['std']
        df[column] = normalizePdSeries(df[column], mean, std)
    return df


def main():
    try :
        with open('training.json', 'r') as data_file:
            data = json.load(data_file)
        entry = ds.load_csv('datasets/dataset_test.csv')
        parseEntry = entry.drop(columns=['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday'])
        # print(parseEntry)
        params = open('all_parameters.json')
        with open('all_parameters.json', 'r') as params_file:
            params = json.load(params_file)
        params = pd.DataFrame(params['data'])
        entryNormalized = extractAndPrepareNumericalDatas(parseEntry, params)
        print(entryNormalized)

    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()


# Y = x1 * 