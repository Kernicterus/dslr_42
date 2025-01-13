import matplotlib.pyplot as plt
import dstools as ds

HouseColor = {
    'Gryffindor' : 'tab:red',
    'Ravenclaw' : 'b',
    'Hufflepuff' : 'y',
    'Slytherin' : 'tab:green', 
}

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

def scatter_plot(data, feature1, feature2, ax, size):
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
            ax.scatter(x,y, label=house, alpha=0.8, s=size, color=HouseColor[house])

def histogram(data, field, ax):
        cleanData = data.dropna()

        HogwartHouse = {
            'Gryffindor' : cleanData.loc[cleanData['Hogwarts House'] == 'Gryffindor', :],
            'Ravenclaw' : cleanData.loc[cleanData['Hogwarts House'] == 'Ravenclaw', :],
            'Hufflepuff' : cleanData.loc[cleanData['Hogwarts House'] == 'Hufflepuff', :],
            'Slytherin' : cleanData.loc[cleanData['Hogwarts House'] == 'Slytherin', :], 
        }
        for house in HogwartHouse:
            df = HogwartHouse[house]
            serie = df[field]
            ax.hist(serie, bins=20, alpha=0.4, label=house, color=HouseColor[house])

def onClick(event, data):
    if event.inaxes:
        clickedAxes = event.inaxes
        i, j = clickedAxes.indices
        fig_new = plt.figure(figsize=(10, 8))
        ax_new = fig_new.add_subplot(111)
        if i == j:
            histogram(data, features[i], ax_new)
            plt.ylabel('Frequency')
            plt.xlabel(features[i])
        else:
            scatter_plot(data, features[i], features[j], ax_new, 10)
            plt.xlabel(features[i])
            plt.ylabel(features[j])
        handles = [plt.Line2D([0], [0], color=HouseColor[house], marker='o', label=house) for house in HouseColor]
        fig_new.legend(handles=handles, loc='upper right')
        plt.figure(2)
        plt.show()

def main():
    try:
        data = ds.load_csv('datasets/dataset_train.csv')

        fig, axs = plt.subplots(13, 13, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        i = 0
        j = 0
        for i in range(13):
            axs[i][0].set_ylabel(features[i], rotation=0, ha='right', fontsize=8)
            for j in range(13):
                axs[i][j].indices = (i, j)
                if not i == j:
                    scatter_plot(data, features[i], features[j], axs[i][j], 1)
                else:
                    histogram(data, features[i], axs[i][j])
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                if j == 12:
                        axs[12][i].set_xlabel(features[i], rotation=30, ha='right', fontsize=8)
        handles = [plt.Line2D([0], [0], color=HouseColor[house], marker='o', label=house) for house in HouseColor]
        fig.legend(handles=handles, loc='upper right')
        fig.canvas.mpl_connect('button_press_event', lambda event: onClick(event, data))
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()