import dstools as ds
import pandas as pd
import json
import sys
import os

HOUSE = [
        'Gryffindor',
        'Slytherin',
        'Ravenclaw',
        'Hufflepuff',
        ]

DISCRETE_FEATURES = [
        'First Name',
        'Last Name',
        'Best Hand',
        'Year',
        'Month',
        'Day',
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
    try:
        df[['Year', 'Month', 'Day']] = df['Birthday'].str.split('-', expand=True)
        df = df.drop(columns=['Birthday'])
    except Exception:
        pass
    for column in df.columns:
        if column in DISCRETE_FEATURES:
            df[column] = df[column].fillna(trainingParam[column]['mean'])
        else:
            df[column] = df[column].fillna(trainingParam[column]['median'])
    for column in df.columns:
        if column in DISCRETE_FEATURES:
            df[column] = df[column].astype('category').cat.codes
        mean = trainingParam[column]['mean']
        std = trainingParam[column]['std']
        df[column] = normalizePdSeries(df[column], mean, std)
    return df


# Probabilité = 1 / (1 + e^(-z))
# où z = biais + (poids de Best Hand × valeur normalisée de Best Hand) +
# (poids d'Arithmancy × valeur normalisée d'Arithmancy) + ...

# On fait d'abord la somme de tous les produits (poids × valeur) plus le biais, ce qui donne z
# PUIS on applique la fonction sigmoïde : 1 / (1 + e^(-z))

# La fonction sigmoide transforme le resultat en une probabilite entre 0 et 1


def predict(df : pd.DataFrame, weight : pd.DataFrame):
    predictions = pd.Series(index=df.index, dtype='string')
    for index, row in df.iterrows():
        result = pd.Series(dtype='float64')
        for house in HOUSE:
            w = weight[house]
            a = 0
            for feature in row.index:
                a += row[feature] * w[feature]
            z = w['Intercept'] + a
            result[house] = ds.sigmoid(z)
        predictions.loc[index] = ds.maxObj(result)
    return predictions


def main():
    try :
        if len(sys.argv) != 3:
            print('Error : dataset to predict and weight.json needed')
            return 1
        with open(sys.argv[2], 'r') as data_file:
            weight = json.load(data_file)
        entry = ds.load_csv(sys.argv[1])
        parseEntry = entry.drop(columns=['Index', 'Hogwarts House'])
        with open('all_parameters.json', 'r') as params_file:
            params = json.load(params_file)
        params = pd.DataFrame(params['data'])
        entryNormalized = extractAndPrepareNumericalDatas(parseEntry, params)
        result = predict(entryNormalized, weight['data'])

        result.to_csv('houses.csv', header=['Hogwarts House'], index=True)
        compare = entry['Hogwarts House'].eq(result)
        print("Precision = ", (compare.sum() / len(compare)) * 100)
    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()
