import dstools as ds
import pandas as pd

HOUSE = [
        'Gryffindor',
        'Slytherin',
        'Ravenclaw',
        'Hufflepuff',
        ]

def normalizeSeries(column : pd.Series, mean, std):
    variableNormalized = (column - mean) / std
    return variableNormalized

def extractAndPrepareNumericalDatas(df : pd.DataFrame, trainingParam : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    for column in df:
        a = trainingParam.filter
        print(a)
        # print(column)

def main():
    try :
        data = open('training.json')
        entry = ds.load_csv('datasets/dataset_test.csv')
        params = open('all_parameters.json')
        entryNormalized = extractAndPrepareNumericalDatas(entry, params)


    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()


# Y = x1 * 