import cv2
import numpy as np
from sklearn.cluster import KMeans
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import pandas as pd


# Helper function to calculate centroid(mean) of points belonging to a cluster
def computeCentroid(points):
    # Numeber of points in cluster
    num_points = len(points)

    # calculating mean by taking column-wise(axis=0) sum of points and dividing by no. points
    centroid = np.sum(points, axis=0) / num_points
    return centroid


# Helper function to calculate the euclidean distance (distance formula)
def distance(point, centroid):
  # subracting the (mayebe)multi-dimentional points
    difference = point - centroid

  # squaring the difference in accordance to distance formula
    diff_sq = difference * difference

  # taking sum of squares of differnces
    sum_of_sqs = np.sum(diff_sq)
  # finally taking square root to calculate Eucleidian distance
    dist = np.sqrt(sum_of_sqs)

    return dist


# Scratch implementation of Kmeans that returns centroids along with labels of points
def k_means(points, centroids, clusters):
    epochs = 10
    m, n = points.shape
  # creating a numpy array containing zero initially to hold which cluster no. the datapoints belong
    index = np.zeros(m, dtype=int)

    while epochs > 0:
        for i in range(m):
          # intializing to positive infinity
            min_dist = float('inf')
          # for storing cluster index temporily
            temp = None

            for k in range(clusters):
              # making two numpy arrays
                point = points[i]
                centroid = centroids[k]

              # Using helper function for calculating eucliadian distance between point and centroid
                dist = distance(point,centroid)

              # assigning cluster(label) to a point based on the minimum distance of it from centroid of that cluster
                if dist < min_dist:
                    # updating the min_dist
                    min_dist = dist
                    # temp variable for kepping track
                    temp = k
                    # saving the label of the point
                    index[i] = k

      # updating the centroids of cluster based on points assigned to that cluster
        for k in range(clusters):
          # points assigned to cluster
            cluster_points = points[index == k]

            # if points are added to that cluster
            if len(cluster_points) > 0:
              # centroids are calculated along axis 0 using helper function
                centroids[k] = computeCentroid(cluster_points)

        epochs -= 1

    return centroids, index

# wrapper function for scratch implementation of K-means clustering
def mykmeans(X, k):
    # Initialize the cenytroids by randomly selecting k datapoints from X
    centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
    centroids, labels = k_means(X, centroids, k)
    return centroids, labels
# Reading the image
img = cv2.imread('test.png')

# Convert the image from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Flatten the image into a 2D array
img_flatend = img_rgb.reshape(-1, 3)

# Converting the image to float data type that would be helpful while scaling
img_flatend = img_flatend.astype(float)

# Scaling the image pixel values in the range of 0 to 1
img_scaled = img_flatend / 255.0

# printing the shapes of image and scaled image
print("Image shape", img.shape)
print("Reshaped Image shape", img_scaled.shape)

ks = [2, 4, 8]

# for printing images comprresed using sklearn Kmeans and Kmeans from scratch
for k in ks:
    # Perform K-means clustering using scratch implementation mykmeans
    centroids, labels = mykmeans(img_scaled, 4)

    # Replacing each pixel value with the centroid of its cluster
    img_compressed = centroids[labels]

    # Reshaping to original shape of image
    img_compressed = img_compressed.reshape(img.shape)

    # Convert the compressed image back to the range 0-255
    img_compressed = img_compressed * 255

    img_compressed = img_compressed .astype(np.uint8)

    img_bgr = cv2.cvtColor(img_compressed, cv2.COLOR_RGB2BGR)
    #from google.colab.patches import cv2_imshow

    # Display the image in a window named 'image'
    print()
    print(f"Image compressed with {k} clusters(colors)")
    cv2.imshow('Image compressed with 4 clusters(colors)', img_bgr)

    # Wait for any key to be pressed and then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from skimage.metrics import structural_similarity as ssim

# Convert the original and compressed images to float type
img_float = img.astype(float)
img_compressed_float = img_bgr.astype(float)

print()
print("---------------------------------------------------------------")
print("")
# Calculate Mean Squared Error (MSE)
mse = np.mean((img_float - img_compressed_float) ** 2)
print(f'MSE: {mse}')

# Calculate Peak Signal-to-Noise Ratio (PSNR)
psnr = cv2.PSNR(img, img_bgr)
print(f'PSNR: {psnr}')

# Calculate Structural Similarity Index (SSIM)
ssim_value = ssim(img, img_compressed, channel_axis=-1)
print(f'SSIM: {ssim_value}')
print("Compression of image for other values of k are coming up next---")
# Define the number of clusters (colors)
ks = [2, 4, 8, 16, 32]

# for printing images comprresed using sklearn Kmeans and Kmeans from scratch
for k in ks:
    # Perform K-means clustering using scratch implementation mykmeans
    centroids, labels = mykmeans(img_scaled, k)

    # Replacing each pixel value with the centroid of its cluster
    img_compressed_scratch = centroids[labels]

    # Reshaping to original shape of image
    img_compressed_scratch = img_compressed_scratch.reshape(img.shape)

    # Convert the compressed image back to the range 0-255
    img_compressed_scratch = img_compressed_scratch * 255

    img_compressed_scratch = img_compressed_scratch .astype(np.uint8)

    # Perform K-means clustering using sklearn
    kmeans = KMeans(n_clusters=k)
    # training the kmeans
    kmeans.fit(img_scaled.reshape(-1, 3))
    # Replacing each pixel with the centroid of its cluster
    img_compressed_sklearn = kmeans.cluster_centers_[kmeans.labels_]

    # Reshaping to original shape of image
    img_compressed_sklearn = img_compressed_sklearn.reshape(img.shape)

    # Convert the compressed image back to the range 0-255 and to uint8 from float
    img_compressed_sklearn = (img_compressed_sklearn * 255).astype(np.uint8)

    # Convert the images back from RGB to BGR
    img_bgr_scratch = cv2.cvtColor(img_compressed_scratch, cv2.COLOR_RGB2BGR)
    img_bgr_sklearn = cv2.cvtColor(img_compressed_sklearn, cv2.COLOR_RGB2BGR)

    # Display the compressed images
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr_scratch, cv2.COLOR_BGR2RGB))
    plt.title(f'Scratch KMeans with {k} colors')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_bgr_sklearn, cv2.COLOR_BGR2RGB))
    plt.title(f'Sklearn KMeans with {k} colors')
    plt.axis('off')

    plt.show()



# for printing original images vs images compressed using sklearn
# Loop over the number of clusters
for k in ks:
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img_scaled)

    labels = kmeans.predict(img_scaled)
    centroids = kmeans.cluster_centers_
    # Replace each pixel value with the centroid of its cluster
    img_compressed = centroids[labels]
    img_compressed = img_compressed.reshape(img.shape)

    # Convert the compressed image back to the range 0-255
    img_compressed = (img_compressed * 255)
    img_compressed = img_compressed.astype(np.uint8)

    # Convert the image from RGB to BGR
    img_bgr = cv2.cvtColor(img_compressed, cv2.COLOR_RGB2BGR)

    # Display the original image
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Display the compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f'Compressed Image with {k} colors with Sklearn')
    plt.axis('off')

    plt.show()
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load an image file
img = cv2.imread('test.png')  # replace with your image file path

# Flatten the image and scale the RGB values to the range 0-1
img_flat = img.reshape(-1, 3)
img_flat = img_flat.astype(float)
img_flat = img_flat / 255.0

# the x and y coordinates of each pixel
x, y = np.indices((img.shape[1], img.shape[0]))

w = img.shape[1]
l = img.shape[0]

x_scaled = x.flatten() / w
y_scaled = y.flatten() / l

# Adding the x,y coordinates to the feature vectors using column stack funtion of np
img_with_coords = np.column_stack((img_flat, x_scaled, y_scaled))

# printing the shapes of image and scaled image
print("Image shape", img.shape)
print("Reshaped image shape without space coordinates ", img_flat.shape)
print("Reshaped Image shape along with space coordinates in features", img_with_coords.shape)
print()
print()
# k is the number of clusters
k = 16

# Sklearn K-means clustering with spatial coherence
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(img_with_coords)

# label assignment for each pixel along with spacial coherence
labels = kmeans.predict(img_with_coords)

# Replace each pixel value with the centroid of its cluster

centroids = kmeans.cluster_centers_[:, :3]
img_compressed = centroids[labels]
img_compressed = img_compressed * 255

# Reshape the compressed image to the original shape
img_compressed = img_compressed.reshape(l, w, 3)

# Convert the compressed image back to the range 0-255 and to uint8
img_compressed = img_compressed.astype(np.uint8)

# Convert the image from RGB to BGR for display with OpenCV
img_bgr = cv2.cvtColor(img_compressed, cv2.COLOR_RGB2BGR)

# Sklearn K-means clustering without spatial coherence
kmeans_without_spatial = KMeans(n_clusters=k, n_init=10)
kmeans_without_spatial.fit(img_flat)

# label assignment for each pixel without spacial coherence
labels_without_spatial = kmeans_without_spatial.predict(img_flat)

# Replace each pixel value with the centroid of its cluster
centroids_without_spatial = kmeans_without_spatial.cluster_centers_
img_compressed_without_spatial = centroids_without_spatial[labels_without_spatial]
img_compressed_without_spatial = img_compressed_without_spatial * 255

# Reshape the compressed image to the original shape
img_compressed_without_spatial = img_compressed_without_spatial.reshape(l, w, 3)

# Convert the compressed image back to the range 0-255 and to uint8
img_compressed_without_spatial = img_compressed_without_spatial.astype(np.uint8)

# Convert the image from RGB to BGR for display with OpenCV
img_bgr_without_spatial = cv2.cvtColor(img_compressed_without_spatial, cv2.COLOR_RGB2BGR)

# Display the original image
plt.figure(figsize=(10, 5))

# Display the compressed image without spatial coherence
plt.subplot(1, 2, 1)
plt.imshow(img_bgr_without_spatial)
plt.title(f'Compressed without spatial coherence')
plt.axis('off')

# Display the compressed image with spatial coherence
plt.subplot(1, 2, 2)
plt.imshow(img_bgr)
plt.title(f'Compressed with spatial coherence')
plt.axis('off')

plt.show()

from skimage.metrics import structural_similarity as ssim

# Convert the original and compressed images to float type
img_float = img.astype(float)
img_compressed_float = img_compressed.astype(float)
print("Stats For clustering with spacial coherence")
# Calculate Mean Squared Error (MSE)
mse = np.mean((img_float - img_compressed_float) ** 2)
print(f'MSE: {mse}')

# Calculate Peak Signal-to-Noise Ratio (PSNR)
psnr = cv2.PSNR(img, img_compressed)
print(f'PSNR: {psnr}')

# Calculate Structural Similarity Index (SSIM)
ssim_value = ssim(img, img_compressed, channel_axis=-1)
print(f'SSIM: {ssim_value}')

img_compressed_float_c = img_compressed_without_spatial.astype(float)
print()
print("---------------------------------------------------------------")
print()
print("Stats for clustering without spacial coherence")
# Calculate Mean Squared Error (MSE)
mse = np.mean((img_float - img_compressed_float_c) ** 2)
print(f'MSE: {mse}')

# Calculate Peak Signal-to-Noise Ratio (PSNR)
psnr = cv2.PSNR(img, img_compressed_without_spatial)
print(f'PSNR: {psnr}')

# Calculate Structural Similarity Index (SSIM)
ssim_value = ssim(img, img_compressed_without_spatial, channel_axis=-1)
print(f'SSIM: {ssim_value}')

