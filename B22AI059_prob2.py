from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


# Loading the iris dataset as per question
iris = datasets.load_iris(as_frame=True)

# Extracting only petal length and petal width features
X = iris['data'][['petal length (cm)', 'petal width (cm)']]
y = iris['target']

# Selecting only 'setosa' and 'versicolor' classes by creating a binary mask
bin_mask = iris['target']<2

# X and y with onl setosa and versicolor labels(binary)
X_binary = X[bin_mask]
y_binary = y[bin_mask]

# Normalize the dataset using standardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_binary)

# Create a scatter plot
plt.figure(figsize=(8, 6))

# (x_s,y_s) points of setosa class corresponding to the petal length and petal width
x_s = X_norm[y_binary==0, 0]
y_s = X_norm[y_binary==0, 1]

# (x_v,y_v) points of versicolor class corresponding to the petal length and petal width
x_v = X_norm[y_binary==1, 0]
y_v = X_norm[y_binary==1, 1]

# Data Visualization
# a scatter plot of setosa and versicolor class on the axes
plt.scatter(x_s, y_s, color='red', label='setosa',edgecolor='black')
plt.scatter(x_v, y_v, color='lightgreen', label='versicolor', edgecolor='black')

# fixing the labels and title
plt.xlabel('Petal length (normalized)')
plt.ylabel('Petal width (normalized)')
plt.title('Iris dataset: petal length vs petal width')
plt.legend()

# Show the plot
plt.show()


# Splitting the normalized dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_binary, test_size=0.2, random_state=42)


# Instance of the Linear SVM from sklearn.svm module
svc = LinearSVC() # Object represent a linear SVC

# Training a LinearSVC on the training data
svc.fit(X_train, y_train)

# a Helper function to plot the decision boundary
def plot_decision_boundary(svc, X, y):
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # minimum of petal length
    x_min = X[:, 0].min()
    # a margin of 1
    x_min = x_min -1
    # maximum of petal length
    x_max = X[:, 0].max()
    # a margin of 1
    x_max += 1

    # similarly for y_min and y_max (petal width)
    y_min = X[:, 1].min()
    y_min -= 1

    y_max = X[:, 1].max()
    y_max += 1
    
    # creating 1D arrays for the x and y
    x_range = np.arange(x_min, x_max, h)
    y_range = np.arange(y_min, y_max, h)

    # creating a mesh grid using two 2D arrays 
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # flatting the arrays
    x_grid_flat = x_grid.ravel()
    y_grid_flat = y_grid.ravel()

    # concatenating the x, y column-wise flattened arrays to predict
    x_y_concat = np.c_[x_grid_flat, y_grid_flat]
    # predict the labels pf each point in the grid
    Z = svc.predict(x_y_concat)

    # Reshape the 1D array into a 2D array of same shape as x_grid
    Z = Z.reshape(x_grid.shape)
    
    # plotting filled contour plot
    plt.contourf(x_grid, y_grid, Z, cmap=cmap_light, alpha=0.8)

    # Scatter Plot the data points 
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=['#FF0000', '#00FF00'], edgecolor="black", legend='full')

# Plotting the decision boundary of the model on the training data
plt.figure(figsize=(8, 6))
plot_decision_boundary(svc, X_train, y_train)
plt.title('Decision boundary on the training data')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

# Plotting the decision boundary of the model on the test data
plt.figure(figsize=(8, 6))
plot_decision_boundary(svc, X_test, y_test)
plt.title('Decision boundary on the test data')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

# Calculate accuracy on test data
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print()
print("----------------------------------------------------------")
print()
print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print("------------------------------------------------------------")

from sklearn.datasets import make_moons
# Generate synthetic dataset
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

# Maing a dataframe of the genrerated data
df = pd.DataFrame(data=X, columns=['Feature 1', 'Feature 2'])
df['Target'] = y

# Data Visualization
# Use seaborn to create a scatterplot
sns.scatterplot(data=df, x='Feature 1', y='Feature 2', hue='Target', palette=['red', 'lightgreen'] , edgecolor='black')

# Display the plot
plt.show()

from sklearn.svm import SVC

# Implementing SVC with a linear kernel
svc_lin = SVC(kernel='linear')
# training on sythetic data
svc_lin.fit(X, y)
# plotting the decision boundry using pre-defined Helper function  
plot_decision_boundary(svc_lin, X, y)
plt.title('SVM Decision Boundary (Linear Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Implementing SVC with a polynomial kernel
svc_poly = SVC(kernel='poly')
# training on sythetic data
svc_poly.fit(X, y)
# plotting the decision boundry using pre-defined Helper function  
plot_decision_boundary(svc_poly, X, y)
plt.title('SVM Decision Boundary (Polynomial Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Implementing SVC with an RBF kernel
svc_rbf = SVC(kernel='rbf')
# training on sythetic data
svc_rbf.fit(X, y)
# plotting the decision boundry using pre-defined Helper function  
plot_decision_boundary(svc_rbf, X, y)
plt.title('SVM Decision Boundary (RBF Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.model_selection import GridSearchCV

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the Parameter values that should be searched 
parameter_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}

svm_model = SVC(kernel = 'rbf')

grid = GridSearchCV(svm_model, parameter_grid,cv=5, refit=True, verbose=3)
grid.fit(X_train,y_train)

print()
print("------------------------------------------------")
print()
best_parameters =grid.best_params_
print("Best hyperparameters:", best_parameters)

best_svm_model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
best_svm_model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy*100} %")

# Plot the decision boundary for the RBF kernel SVM with the best hyperparameters
plot_decision_boundary(best_svm_model, X, y)
plt.title('SVM Decision Boundary (RBF Kernel with Best Hyperparameters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
