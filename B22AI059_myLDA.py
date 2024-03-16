import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv, eig

# For Separating the data into class 0 and class 1 
def FilterData(X):

    # Taking the first two columns of the data as features of class 0, 1 based on the label in the third column
    class_0 = []
    class_1 = []
    for row in X:
        # If the label is 0, add the first two columns to class_0
        if row[2] == 0:
            class_0.append(row[:2])
        # If the label is 1, add the first two columns to class_1
        elif row[2] == 1:
            class_1.append(row[:2])
    
    # Convert the lists to numpy arrays
    class_0 = np.array(class_0)
    class_1 = np.array(class_1)

    return class_0, class_1

# Compute the difference of class-wise means as per Task 1 
def ComputeMeanDiff(X):

    #Filter the data to get class 0 and class 1
    class_0,class_1 = FilterData(X)
    # Compute the mean of each class
    mean_class1 = np.mean(class_1, axis=0)
    mean_class0 = np.mean(class_0, axis=0)

    # Compute the absolute difference of class-wise means
    mean_diff = abs(mean_class1 - mean_class0)
    return mean_diff

# Compute the Total within-class scatter matrix SW as per Task 1
def ComputeSW(X):
    #Filter the data to get class 0 and class 1
    class_0,class_1 = FilterData(X)

    # Compute the mean of each class
    u0 = np.mean(class_0, axis=0)
    u1 = np.mean(class_1, axis=0)
    
    # Compute the scatter matrix for each class

    # S0 is the scatter matrix for class 0
    S0 = np.zeros((class_0.shape[1], class_0.shape[1]))

    # For each sample in class 0
    for x in class_0:
        # Compute the difference between the sample and the mean of class 0
        difference = x - u0
        # Reshape the difference to 2D
        difference = difference.reshape(-1, 1)  
        S0 += np.dot(difference, difference.T)  # Compute the dot product of diff with its transpose and add to S0

    # S1 is the scatter matrix for class 1
    S1 = np.zeros((class_1.shape[1], class_1.shape[1]))

    # For each sample in class 1
    for x in class_1:
        # Compute the difference between the sample and the mean of class 1
        difference = x - u1
        # Reshape the difference to 2D
        difference = difference.reshape(-1, 1)
        # Compute the dot product of diff with its transpose and add to S1 
        S1 += np.dot(difference, difference.T) 


    # Total Within Scatter Matrix SW is the sum of S0 and S1   
    SW = S0 + S1
    return SW

# Compute the Between class Scatter Matrix SB as per Task 1
def ComputeSB(X):
    #Filter the data to get class 0 and class
    class_0,class_1 = FilterData(X)
    # Compute the difference of class-wise means using the function ComputeMeanDiff 
    mean_diff = ComputeMeanDiff(X)
    # Reshape the mean_diff to 2D
    mean_diff = mean_diff.reshape(-1, 1)  
    # Compute the dot product of mean_diff with its transpose to get the scatter matrix SB
    SB = np.dot(mean_diff, mean_diff.T)  
    return SB

# Compute the LDA projection vector as per Task 1
def GetLDAProjectionVector(X):
    # Compute the Total within-class scatter matrix SW using the function ComputeSW
    SW = ComputeSW(X)
    # Compute the Between class Scatter Matrix SB using the function ComputeSB
    SB = ComputeSB(X)
    # Compute the inverse of SW
    SW_inv = inv(SW)
    # Compute the eigenvalues and eigenvectors of the dot product of SW_inv and SB
    eigvals, eigvecs = eig(np.dot(SW_inv, SB))
    # Get the index of the maximum eigenvalue
    max_eig_idx = np.argmax(eigvals)
    # Get the eigenvector corresponding to the maximum eigenvalue as per the Algorithm
    w = eigvecs[:,max_eig_idx]
    return w

# Project a 2D point onto the LDA projection vector as per Task 1
def project(x, y, w):
    # Create a 2D point using the input x and y
    point = np.array([x, y])
    w_norm = w / np.linalg.norm(w)  # Normalize w
    projection_length = np.dot(point, w_norm)  # Project point onto w_norm
    projection_point = projection_length * w_norm  # Scale projection length by w_norm
    return projection_point

# Main function
def main():
    # Read the data from the file data.csv
    X = np.empty((0, 3))
    with open('data.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for sample in csvFile:
            X = np.vstack((X, [float(i) for i in sample]))


    # Print the data and its shape    
    print(X)
    print(X.shape)

    # Menu-driven program
    
    print("Options:")
    print("1. Compute the difference of class-wise means")
    print("2. Compute the Total within-class scatter matrix SW")
    print("3. Compute the Between class Scatter Matrix SB")
    print("4. Compute the LDA projection vector")
    print("5. Project a 2-dimensional point onto the LDA projection vector")
    print("0. Exit")
    print()
    while True:
        # Display the options
        opt = int(input("Input your option (1-5), press 0 to exit: "))
        # If the user chooses option 0, exit the program
        if opt == 0:
            print("Exiting...")
            break
        #Match the option chosen by the user and perform the corresponding operation   
        match opt:
            case 1:
                meanDiff = ComputeMeanDiff(X)
                print("Difference of class-wise means: ", meanDiff)

            case 2:
                SW = ComputeSW(X)
                print("Total within class Scatter matrix SW: \n", SW)

            case 3:
                SB = ComputeSB(X)
                print("Between class Scatter Matrix SB: \n", SB)

            case 4:
                w = GetLDAProjectionVector(X)
                print("Projection Vector:  ", w)

            case 5:
                x = float(input("Input x dimension of a 2-dimensional point: "))
                y = float(input("Input y dimension of a 2-dimensional point: "))
                w = GetLDAProjectionVector(X)
                print("Projected Point:", project(x, y, w))

            
if __name__ == "__main__":
    main()
