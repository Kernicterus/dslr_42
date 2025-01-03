import dstools as ds
import pandas as pd
import math

def isNum(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def columnParser(column: pd.Series):
    try:
        i = 0
        for value in column:
            if (isNum(value) == True):
                if not pd.isna(value):
                    i += 1
            else:
                return -1
        return i
    except Exception as e:
        print(f"Error: {e}")

def mean(column: pd.Series):
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += value
            size += 1
    return sum / size

def std(column: pd.Series, moyenne):
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += (moyenne - value) ** 2
            size += 1
    return math.sqrt(sum / size)

def min(column: pd.Series):
    min = column[0]
    for value in column:
        if not pd.isna(value):
            if value < min:
                min = value
    return min

def max(column: pd.Series):
    max = column[0]
    for value in column:
        if not pd.isna(value):
            if value > max:
                max = value
    return max

def main():
    try :
        data = ds.load_csv('datasets/dataset_train.csv')
        print(data.columns)
        column = data['Transfiguration']

        moy = mean(column)
        count = columnParser(column)

        print('count ', count)
        print('mean ', moy)
        print('std ', std(column, moy))
        print('min', min(column))
        print('max', max(column))

        print('PANDAS')
        print(column.describe())

        # for column in data.columns:
        #     print(column)
        #     count = columnParser(data[column])
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()