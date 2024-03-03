#importing modules and pakages
from time import time
import logging
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from collections import Counter

#load the data into numpy arrays
lfw_people =fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#Find out shape infomration about the images to help with plotting them
n_samples, h, w=lfw_people.images.shape

np.random.seed(42)

#loading the dataset into X and y
X = lfw_people.data
y = lfw_people.target

#For machine learning we use the data directly
n_features = X.shape[1]
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

#Printing the shape of the dataset
print ("Total dataset size:")
print ("n_samples: %d" % n_samples)
print ("n_features: %d" % n_features)
print ("n_classes: %d" % n_classes)
# classes = target_names
print ("Classes: %s" % target_names)

# Plot the first 10 images for visualization
fig, ax = plt.subplots(2, 5, figsize=(15, 8))

# Adjust the subplots
for i, axi in enumerate(ax.flat):
    # Display an image at the i-th position
    axi.imshow(lfw_people.images[i], cmap='gray')
    axi.set_xticks([])  # Remove x-axis ticks
    axi.set_yticks([])  # Remove y-axis ticks
    # Add the target names as the title
    axi.set_xlabel(lfw_people.target_names[lfw_people.target[i]], fontsize=14, labelpad=10)  # Increase font size and add padding

plt.tight_layout()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Standardize the dataset
scaler = StandardScaler()
# Apply transform to both the training set and the test set.
# Fit on training set only.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform PCA without reducing the number of components
pca = PCA(whiten=True, random_state=42)
# Fit the PCA model to the training data
pca.fit(X_train)

# Calculate cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 7))
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choose n_components such that 95% variance is retained
n_components = np.where(cumulative_explained_variance > 0.95)[0][0]
# printing a line
print()
print(f"Choosing {n_components} components")

# Perform PCA with the chosen number of components
pca = PCA(n_components=n_components, whiten=True, random_state=42)
# Fit the PCA model to the training data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# implementing a KNN classifier
class KNN:
    # Initialize the KNN classifier with the number of neighbors
    def __init__(self, k):
        self.k = k
    # Train the classifier using the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    # Predict the classes of the test set
    def predict(self, X):
        predicted_labels = [self.predict2(x) for x in X]
        return np.array(predicted_labels)
    # Helper function to predict the most common label for a sample
    def predict2(self, x):
        # Initialize a list to store the distances
        distances = []
        # Compute distances between x and all examples in the training set
        for x_train in self.X_train:
          #Eucleidian distance formula
            distance = np.sqrt(np.sum((x - x_train)**2))
            distances.append(distance)

        # Get the indices of the sorted distances
        sorted_indices = np.argsort(distances)

        # Get the indices of the first k neighbors
        k_indices = sorted_indices[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Count the occurrences of each label
        label_counts = {}
        for label in k_nearest_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        # Find the label with the most occurrences
        most_common_label = max(label_counts, key=label_counts.get)

        # Return the most common label
        return most_common_label

#To find the optimal number of neighbors (k) that gives the maximum accuracy
# Range of k values for trial
k_range = range(1, 20)

# List to store the accuracy for each value of k
accuracies = []

# var to store value of k corresponding to max accuracy
best_k = 0
max_accuracy = 0

for k in k_range:
    # Create a KNN classifier with the current number of neighbors
    knn = KNN(k=k)

    # Train the classifier
    knn.fit(X_train_pca, y_train)

    # Predict the classes of the test set
    y_pred = knn.predict(X_test_pca)

    # Calculate accuracy from scratch
    correct_predictions = np.sum(y_pred == y_test)
    accuracy = correct_predictions / len(y_test)
    accuracies.append(accuracy)

    # Update best_k if the current accuracy is greater than the maximum accuracy
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_k = k

# Plot accuracy as a function of k
plt.figure(figsize=(10, 7))
plt.plot(k_range, accuracies)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.show()

print("best k =",best_k)

# Create a KNN classifier by training it for best k
knn = KNN(best_k)

# Train the classifier using the PCA-transformed training data
knn.fit(X_train_pca, y_train)

# Predict the classes of the test set
y_pred = knn.predict(X_test_pca)

# Calculate and report accuracy
correct_predictions = np.sum(y_pred == y_test)
accuracy = correct_predictions / len(y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Create a new figure with a specified size
fig = plt.figure(figsize=(10, 5))

# For each of the first 10 Eigenfaces
for i in range(10):
    # Create a subplot in a 2x5 grid at i+1 th postion
    ax = fig.add_subplot(2, 5, i+1)
    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Display the Eigenface
    ax.imshow(pca.components_[i].reshape(h, w), cmap='gray')
    # Set the title for the subplot
    ax.set_title('Eigenface ' + str(i+1))

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Display the plot
plt.show()

# Range of n_components to try and calculate accuaracies
print("Experimenting with diffrent n_components values")
n_components_range = [50, 100, 150, 200, 250]

# List to store the accuracy for each value of n_components
accuracies = []

for n_components in n_components_range:
    # Perform PCA with the current number of components
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    # Fit the PCA model to the training data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # create a KNN classifier using KNN class
    knn = KNN(best_k)
    # Train the classifier using the PCA_tranformed training data
    knn.fit(X_train_pca, y_train)

    # Predict the classes of the test set
    y_pred = knn.predict(X_test_pca)

    # Calculate and print accuracy corresponding to n_component value
    correct_predictions = np.sum(y_pred == y_test)
    accuracy = correct_predictions / len(y_test)
    accuracies.append(round(accuracy* 100,2))

    print(f"n_components = {n_components}, Accuracy = {accuracy * 100:.2f}%")

# A table to show n_components varying with accuracy
from tabulate import tabulate

# Convert the results dictionary to a list of lists
results_list = list(zip(n_components_range, accuracies))

# Create a table with 'fancy_grid' style
table = tabulate(results_list, headers=['n_components', 'Accuracy'], tablefmt='fancy_grid', floatfmt=".4f")

# Print the table
print(table)

# Plot accuracy as a function of n_components
plt.figure(figsize=(10, 7))
plt.plot(n_components_range, accuracies)
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy')
plt.show()

