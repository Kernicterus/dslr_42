# DSLR - Data Science: Logistic Regression


## Overview
This **team project** from 42 School implements a logistic regression algorithm to replace the Sorting Hat from Harry Potter. The algorithm predicts which Hogwarts House a student belongs to by analyzing their exam scores and other attributes. Based on these features, the model determines whether students should be sorted into Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.

## Features
- Data visualization tools to explore correlations between features
- Implementation of logistic regression from scratch
- Prediction model that replaces the Sorting Hat
- Detailed descriptive statistics of the dataset

## Implementation Details

### Tools and Methods
- **Descriptive Statistics**: Custom implementation of statistical functions (mean, std, percentile)
- **Data Visualization**:
  - Histograms to visualize feature distributions by house
  - Scatter plots to examine relationships between features
  - Pair plots to analyze multiple feature correlations
- **Machine Learning**:
  - Logistic regression implemented **without** using scikit-learn
  - One-vs-all technique for multi-class classification
  - Gradient descent optimization algorithm
  - Sigmoid activation function

### Project Structure
- **describe.py**: Provides statistical analysis of the dataset
- **histogram.py**: Generates histograms to visualize data distribution
- **pair_plot.py**: Creates pair plots to visualize relationships between features
- **scatter_plot.py**: Creates scatter plots for two selected features
- **logreg_train.py**: Implements the logistic regression training algorithm
- **logreg_predict.py**: Uses the trained model to predict house assignments
- **dstools.py**: Contains utility functions for data processing and analysis

## Technical Concepts Learned
- **Data Preprocessing**:
  - Handling missing values
  - Feature normalization
  - Data transformation
- **Statistical Analysis**:
  - Implementation of statistical measures from scratch
  - Data exploration and visualization
- **Machine Learning**:
  - Logistic regression algorithm implementation
  - One-vs-all classification technique
  - Gradient descent optimization
  - Model evaluation and accuracy assessment

## Usage
1. **Setup the environment**:
   ```bash
   bash setup-venv.sh
   ```

2. **Explore the data**:
   ```bash
   python describe.py
   python histogram.py
   python pair_plot.py
   ```

3. **Train the model**:
   ```bash
   python logreg_train.py datasets/dataset_train.csv
   ```

4. **Make predictions**:
   ```bash
   python logreg_predict.py datasets/dataset_test.csv training.json
   ```

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib

All dependencies can be installed using:
```bash
pip install -r requirements.txt
```

## Acknowledgments

I would like to sincerely thank School 42 for this enriching learning opportunity, as well as my teammate [Armand Aranger](https://github.com/ArmandI0) for his valuable collaboration throughout this project.