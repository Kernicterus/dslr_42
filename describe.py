import dstools as ds
import pandas as pd

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

def main():
    try :
        data = ds.load_csv('datasets/dataset_train.csv')

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

        for column in data.columns:

            status = columnParser(data[column])
            if not status == -1:
                tab["field"].append(column)
                tab["count"].append(status)
                tab["mean"].append(ds.mean(data[column]))
                tab["std"].append(ds.std(data[column]))
                tab["min"].append(min(data[column]))
                tab["25%"].append(ds.percentile(data[column], 0.25))
                tab["50%"].append(ds.percentile(data[column], 0.50))
                tab["75%"].append(ds.percentile(data[column], 0.75))
                tab["max"].append(max(data[column]))
        df = pd.DataFrame(tab).transpose()
        df.columns = [""]*len(df.columns)
        print(df)
    except Exception as e:
        print(f"Error: {e}")


if __name__== "__main__":
    main()
