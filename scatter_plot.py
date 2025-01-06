import matplotlib.pyplot as plt
import dstools as ds

def scatter_plot(data, feature1, feature2):
        cleanData = data.dropna()
        HogwartHouse = {
            'Gryffindor' : cleanData.loc[cleanData['Hogwarts House'] == 'Gryffindor', :],
            'Ravenclaw' : cleanData.loc[cleanData['Hogwarts House'] == 'Ravenclaw', :],
            'Hufflepuff' : cleanData.loc[cleanData['Hogwarts House'] == 'Hufflepuff', :],
            'Slytherin' : cleanData.loc[cleanData['Hogwarts House'] == 'Slytherin', :], 
        }
        for house in HogwartHouse:
            df = HogwartHouse[house]
            x = df[feature1]
            y = df[feature2]
            plt.scatter(x,y, label=house, alpha=0.8, s=20)


        plt.legend()
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

def main():
    try:
        data = ds.load_csv('datasets/dataset_train.csv')
        feature1 = 'Care of Magical Creatures'
        feature2 = 'Potions'
        scatter_plot(data, feature1, feature2)

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()