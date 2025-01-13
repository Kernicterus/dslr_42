import matplotlib.pyplot as plt
import dstools as ds


def histogram(data, field):
        cleanData = data.dropna()

        Gryffindor = cleanData.loc[cleanData['Hogwarts House'] == 'Gryffindor', :]
        Ravenclaw = cleanData.loc[cleanData['Hogwarts House'] == 'Ravenclaw', :]
        Hufflepuff = cleanData.loc[cleanData['Hogwarts House'] == 'Hufflepuff', :]
        Slytherin = cleanData.loc[cleanData['Hogwarts House'] == 'Slytherin', :]

        plt.hist(Gryffindor[field], bins=20, alpha=0.4, label='Gryffindor')
        plt.hist(Ravenclaw[field], bins=20, alpha=0.4, label='Ravenclaw')
        plt.hist(Hufflepuff[field], bins=20, alpha=0.4, label='Hufflepuff')
        plt.hist(Slytherin[field], bins=20, alpha=0.4, label='Slytherin')
        plt.xlabel(field)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def main():
    try:
        data = ds.load_csv('datasets/dataset_train.csv')
        field = 'Care of Magical Creatures'
        histogram(data, field)

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()