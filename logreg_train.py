import dstools as ds
import json
import numpy as np
import pandas as pd
import sys
import os

ITERATION = 100

TRAINING_FEATURES = [
        'First Name',
        'Last Name',
        'Best Hand',
        'Birthday',
        'Arithmancy',
        'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Divination',
        'Muggle Studies',
        'Ancient Runes',
        'History of Magic',
        'Transfiguration',
        'Potions',
        'Care of Magical Creatures',
        'Charms',
        'Flying',
        ]

def checkArgs(args : list) -> bool:
    """
    Validates command line arguments for the training script.
    
    Parameters:
        args (list): Command line arguments list from sys.argv
        
    Returns:
        bool: True if all validations pass, False otherwise
        
    Checks:
        - Correct number of arguments
        - CSV file extension
        - File existence
        - File readability
        - Non-empty file content
    """
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

def saveDatas(weights : pd.Series, params : pd.DataFrame):
    """
    Saves model weights and parameters to JSON files.
    
    Creates two files:
        - training.json: Contains the weights for each Hogwarts house classifier
        - all_parameters.json: Contains the parameters used for data normalization
    
    Parameters:
        weights (pd.Series): Model weights for each feature across all houses
        params (pd.DataFrame): Statistical parameters used for data normalization
    """
    json_structure = {"data": weights.to_dict()}
    with open("training.json", "w") as file:
        json.dump(json_structure, file, indent=4)
    json_structure = {"data": params.to_dict()}
    with open("all_parameters.json", "w") as file:
        json.dump(json_structure, file, indent=4)


def updWeights(weights : pd.Series, dfNormalized : pd.DataFrame, alpha: float, results: pd.Series, nbEl : int) -> pd.Series:
    """
    Updates weights using gradient descent for logistic regression.
    
    Implementation of gradient descent update step for a single iteration:
    1. Calculates predictions using current weights
    2. Computes the error between predictions and actual values
    3. Updates weights based on the error and learning rate
    
    Parameters:
        weights (pd.Series): Current weights for each feature
        dfNormalized (pd.DataFrame): Normalized feature data
        alpha (float): Learning rate for gradient descent
        results (pd.Series): Target values (0 or 1) for current house
        nbEl (int): Number of training examples
        
    Returns:
        pd.Series: Updated weights after one iteration of gradient descent
        
    Raises:
        ValueError: If dimensions of column and error don't match
    """
    newWeights = weights.copy()
    estimatedResults = pd.Series([0] * len(results))
    estimatedResults = dfNormalized.apply(lambda x: ds.predictionH0(weights, x), axis=1)
    error = estimatedResults - results
    sumErrorBias = error.sum()
    for index, value in weights.items():
        if index == 'Intercept':
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
    Performs gradient descent to find optimal weights for logistic regression.
    
    This function:
    1. Initializes weights to zero
    2. Iteratively updates weights using the updWeights function
    3. Continues for a specified number of iterations
    
    Parameters:
        dfNormalized (pd.DataFrame): Normalized feature data with intercept term
        alpha (float): Learning rate for gradient descent
        results (pd.Series): Target values (0 or 1) for current house
        iteration (int): Number of iterations to run gradient descent
        
    Returns:
        pd.Series: Optimized weights for logistic regression model
    """
    weights = pd.Series([0.0] * len(dfNormalized.columns), index=dfNormalized.columns)
    for iteration in range(iteration):
        weights = updWeights(weights, dfNormalized, alpha, results, len(dfNormalized))
    return weights

def prepareResults(df : pd.DataFrame) -> pd.DataFrame:
    """
    Prepares target values for one-vs-all logistic regression classifiers.
    
    Creates binary classifiers for each Hogwarts house:
    - For each house, creates a column where students in that house get a 1, others get 0
    - Implements the "one-vs-all" technique for multi-class classification
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'Hogwarts House' column
        
    Returns:
        pd.DataFrame: DataFrame with binary columns for each house (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)
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
    """
    Main function that executes the logistic regression training pipeline.
    
    Steps:
    1. Validate command line arguments
    2. Load and preprocess the dataset
    3. Extract and normalize numerical and categorical features
    4. Add intercept term and prepare training data
    5. Create one-vs-all target values for each house
    6. Train logistic regression model for each house
    7. Save model weights and parameters
    8. Evaluate model accuracy on training data
    
    Returns:
        int: 1 if error in arguments, None otherwise
    
    Side effects:
        - Creates training.json and all_parameters.json files
        - Prints model accuracy
    """
    try :
        if checkArgs(sys.argv) == False:
            print("Usage: python logreg_train.py <dataset.csv>")
            return 1
        
        # step 1 : load the dataset
        df = pd.read_csv(sys.argv[1])
        df = df[['Index'] + ['Hogwarts House'] + TRAINING_FEATURES]

        # step 2 : drop the rows with missing values in the Hogwarts House column
        df = df.dropna(subset=['Hogwarts House'])

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = ds.extractAndPrepareNumericalDatas(df)

        # step 4 : extraction, numerization, filling missing values (MEAN) and standardization of discrete datas
        discreteDatas, discreteDatasParams = ds.extractAndPrepareDiscreteDatas(df)

        # step 5 : regroup the datas and add the intercept
        dfWithIntercept = pd.concat([pd.Series([1] * len(df), name='Intercept'), discreteDatas, normalizedDatas], axis=1)
        params =pd.concat([discreteDatasParams, numDatasParams], axis=1)

        # step 6 : prepare the results for each classifier (0 or 1) : one vs all technique
        results = prepareResults(df)

        # step 7 : calculate the weights for each classifier
        weightsGryff = gradiantDescent(dfWithIntercept, 0.1, results['Gryffindor'], ITERATION)
        weightsSlyth = gradiantDescent(dfWithIntercept, 0.1, results['Slytherin'], ITERATION)
        weightsRaven = gradiantDescent(dfWithIntercept, 0.1, results['Ravenclaw'], ITERATION)
        weightsHuffl = gradiantDescent(dfWithIntercept, 0.1, results['Hufflepuff'], ITERATION)
        weights = pd.concat([weightsGryff, weightsSlyth, weightsRaven, weightsHuffl], axis=1)
        weights.columns = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

        # step 8 : save the weights and the parameters
        saveDatas(weights, params)

        # # TESTING -----------------------------------------------------
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
        precision = trueResults.sum() / len(trueResults)
        print(f"Accuracy : {precision}")

    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()