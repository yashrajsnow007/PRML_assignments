# test.py
import numpy as np
import pandas as pd

# Function to load data from a file into a dataframe and Convert the dataframe to a numpy array
def load_data(file_name):
    df = pd.read_csv(file_name, skiprows=1, header=None, sep=' ')
    return df.values

# Load the testing data into a dataframe from test.txt and Convert the dataframe to a numpy array
X_test = load_data('B22AI059_test.txt')

# Load the actual labels into a dataframe from y_test.txt and Convert the dataframe to a numpy array
y_test = load_data('B22AI059_y_test.txt')

# Load the weights and bias from .npy files that were made in train.py
weights = np.load('weights.npy')
bias = np.load('bias.npy')

# Standard Normalize the testing data using the mean and standard deviation of the testing data
mean = np.mean(X_test, axis=0)
std = np.std(X_test, axis=0)
X_test = (X_test - mean) / std

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

# Define the predict function to predict the labels of the testing data
def predict(X):
    linear_output = np.dot(X, weights) + bias
    predicted_y = unit_step_func(linear_output)
    return predicted_y

# Predict the labels of the testing data using the predict function
y_pred = predict(X_test)

# Print the original and predicted labels of the testing data after flattening the arrays to look good while printing 
print(f"original:  {y_test.flatten()}")
print(f"predicted:  {y_pred.flatten()}")

# Calculate accuracy as the number of coorect predictions divided by the total number of predictions
accuracy = np.sum(y_test.flatten() == y_pred.flatten()) / (y_test.size)

print(f"Accuracy for 80-20 split is: {100 * accuracy} %")
print("Accuracy reports for 70,50,20 percent of training data has been pasted on the report.pdf" +"\n"+ " You can refer to the Task 3 of colab .ipynb file in this folder for the code of various training sizes.")
