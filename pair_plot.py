import matplotlib.pyplot as plt
import dstools as ds

def scatter_plot(data, feature1, feature2, ax):
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
            ax.scatter(x,y, label=house, alpha=0.8, s=1)

def histogram(data, field, ax):
        cleanData = data.dropna()

        Gryffindor = cleanData.loc[cleanData['Hogwarts House'] == 'Gryffindor', :]
        Ravenclaw = cleanData.loc[cleanData['Hogwarts House'] == 'Ravenclaw', :]
        Hufflepuff = cleanData.loc[cleanData['Hogwarts House'] == 'Hufflepuff', :]
        Slytherin = cleanData.loc[cleanData['Hogwarts House'] == 'Slytherin', :]

        ax.hist(Gryffindor[field], bins=20, alpha=0.4, label='Gryffindor')
        ax.hist(Ravenclaw[field], bins=20, alpha=0.4, label='Ravenclaw')
        ax.hist(Hufflepuff[field], bins=20, alpha=0.4, label='Hufflepuff')
        ax.hist(Slytherin[field], bins=20, alpha=0.4, label='Slytherin')

def main():
    try:
        data = ds.load_csv('datasets/dataset_train.csv')
        features = ['Arithmancy',
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
                   'Flying']

        fig, axs = plt.subplots(13, 13, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # line, = ax.plot(data[features[0]])
        i = 0
        j = 0
        for i in range(13):
            axs[i][0].set_ylabel(features[i], rotation=0, ha='right', fontsize=8)
            for j in range(13):
                if not i == j:
                    scatter_plot(data, features[i], features[j], axs[i][j])
                else:
                    histogram(data, features[i], axs[i][j])
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                if j == 12:
                        axs[12][i].set_xlabel(features[i], rotation=30, ha='right', fontsize=8)
        feature1 = 'Care of Magical Creatures'
        feature2 = 'Arithmancy'
        # scatter_plot(data, feature1, feature2)
        handles = [plt.Line2D([0], [0], color=c, marker='o', label=l) 
                for c, l in zip(['r', 'y', 'b', 'g'], 
                                ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])]
        fig.legend(handles=handles, loc='upper right')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()