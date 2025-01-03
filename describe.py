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

def std(column: pd.Series):
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += (mean(column) - value) ** 2
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

def q1(column: pd.Series):
    try:
        cleanData = column.dropna()
        sortedData = cleanData.sort_values()
        n = len(sortedData)
        index = (n - 1) * 0.25
        if index.is_integer():
            return sortedData.iloc[index]
        lowerIndex = int(index)
        upperIndex = lowerIndex + 1

        lowerValue = sortedData.iloc[lowerIndex]
        upperValue = sortedData.iloc[upperIndex]

        fraction = index - lowerIndex
        return lowerValue + fraction * (upperValue - lowerValue)

    except Exception as e:
        print(f"Error: {e}")

def mediane(column: pd.Series):
    try:
        cleanData = column.dropna()
        sortedData = cleanData.sort_values()
        n = len(sortedData)
        index = (n - 1) * 0.5
        if index.is_integer():
            return sortedData.iloc[index]
        lowerIndex = int(index)
        upperIndex = lowerIndex + 1

        lowerValue = sortedData.iloc[lowerIndex]
        upperValue = sortedData.iloc[upperIndex]

        fraction = index - lowerIndex
        return lowerValue + fraction * (upperValue - lowerValue)

    except Exception as e:
        print(f"Error: {e}")

def q4(column: pd.Series):
    try:
        cleanData = column.dropna()
        sortedData = cleanData.sort_values()
        n = len(sortedData)
        index = (n - 1) * 0.75
        if index.is_integer():
            return sortedData.iloc[index]
        lowerIndex = int(index)
        upperIndex = lowerIndex + 1

        lowerValue = sortedData.iloc[lowerIndex]
        upperValue = sortedData.iloc[upperIndex]

        fraction = index - lowerIndex
        return lowerValue + fraction * (upperValue - lowerValue)

    except Exception as e:
        print(f"Error: {e}")

def main():
    try :
        tab = {
            "field" : [],
            "count" : [],
            "mean" : [],
            "std" : [],
            "min" : [],
            "25%" : [],
            "50%" : [],
            "75%" : [],
            "max" : [],
        }
        df = pd.DataFrame(tab).transpose()
        data = ds.load_csv('datasets/dataset_train.csv')
        print(data.columns)
        column = data['Arithmancy']

        moy = mean(column)
        count = columnParser(column)

        print('count ', count)
        print('mean ', moy)
        print('std ', std(column))
        print('min', min(column))
        print('max', max(column))
        print('q1', q1(column))
        print('medine', mediane(column))
        print('q4', q4(column))

        print('PANDAS')
        print(column.describe())
        print(data.describe())


        for column in data.columns:
            tab["field"].append(column)
            count = columnParser(data[column])
        print(df)
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()