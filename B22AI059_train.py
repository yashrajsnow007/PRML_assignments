import numpy as np
import pandas as pd

# Load the training data into a dataframe from train.txt and Convert the dataframe to a numpy array
df_train = pd.read_csv('B22AI059_train.txt', skiprows=1, header=None, sep=' ')
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

# Calculate the mean and standard deviation of the training data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Standard Normalize the training data using the mean and standard deviation of the training data
X_train = (X_train - mean) / std

# Define the unit step function as activation function
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

# Create a Perceptron class 
class Perceptron:
    # Initialize the Perceptron class with learning rate, number of iterations, and activation function
    def __init__(self,learning_rate,n_iters):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = np.zeros(X_train.shape[1])
        self.bias = 0
    
    def fitter(self, X, y):

        # Get the number of samples and features
        n_samples ,n_features = X.shape

        #init parameters and initialize weights and bias to random values
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        #convert labels to {0, 1} 
        y_=np.where(y>0, 1, 0)

        # Train the perceptron for n_iters iterations using the training data 
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

# Create an instance of the Perceptron class 
p = Perceptron(learning_rate=0.01, n_iters=1000)
# Train the perceptron using the training data
p.fitter(X_train, y_train)

# Save the weights and bias to .npy files to be used in test.py
np.save('weights.npy', p.weights)
np.save('bias.npy', p.bias)

# Print the weights and bias to check if they are saved correctly
print(f"Weights: {p.weights}")
print(f"Bias: {p.bias}")
