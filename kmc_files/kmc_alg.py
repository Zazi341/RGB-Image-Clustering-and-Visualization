import numpy as np  # to store all the data
import random  # to initiate 3 random starting points


def three_random_coords(mat):
    """
Generate 3 coordinates which will become the starting points for the first iteration of the algorithm
:param mat the matrix containing the data to generate indexes in range
:return 3 random generated numbers between 0 and len(mat) - 1
    """

    # Generate 3 unique random numbers between 0 and len(mat) - 1
    random_numbers = random.sample(range(len(mat)), 3)  # select
    return random_numbers


def k_means_ready(dataset, k=3):
    """Initialize centroids using a basic k-means++ approach."""
    # Randomly choose the first centroid
    centroids = [dataset.iloc[random.randint(0, len(dataset) - 1)][['Red', 'Green', 'Blue']].tolist()]

    # Choose the remaining centroids
    for _ in range(1, k):
        distances = []
        for i in range(len(dataset)):
            row = dataset.iloc[i][['Red', 'Green', 'Blue']].tolist()
            min_distance = min([np.linalg.norm(np.array(row) - np.array(c)) for c in centroids])
            distances.append(min_distance ** 2)
        # Select the next centroid with probability proportional to distance squared
        probabilities = np.array(distances) / sum(distances)
        next_centroid_index = np.random.choice(len(dataset), p=probabilities)
        centroids.append(dataset.iloc[next_centroid_index][['Red', 'Green', 'Blue']].tolist())

    return centroids


def calculate_distance(row, reference_point):
    """
    Calculate Euclidean distance between a dataset row and a reference point.
    :param row: a row from the dataset containing RGB values.
    :param reference_point: a list of RGB values to calculate distance from.
    :return: the Euclidean distance between the two points.
    """
    return np.sqrt((row['Red'] - reference_point[0]) ** 2 +
                   (row['Green'] - reference_point[1]) ** 2 +
                   (row['Blue'] - reference_point[2]) ** 2)


def split_and_update_clusters(dataset, centroids):
    """
    assign points to the nearest centroid and update centroids iteratively until convergence.
    :param dataset: the dataset containing the RGB values.
    :param centroids: the current centroids.
    :return: A tuple containing the updated clusters and distances.
    """

    distance_arr = []  # array that will hold all the distances between each point in the cluster and its centroid
    while True:
        clusters = []

        # Assign each point to the nearest centroid
        for i in range(len(dataset)):
            distances = [calculate_distance(dataset.iloc[i], centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters.append(cluster_index)
            distance_arr.append(distances[cluster_index])

        # Convert clusters list to a NumPy array
        clusters = np.array(clusters)

        # Calculate new centroids
        new_centroids = []
        for i in range(3):
            new_centroid = calculate_means(dataset, clusters, i)
            new_centroids.append(new_centroid)

        # Check for convergence (i.e., centroids do not change)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return [clusters, distance_arr]


def calculate_means(dataset, cluster, k):
    """
    Calculate the mean RGB values for a given cluster.
    :param dataset: the dataset of NameRGB.
    :param cluster: an array containing the cluster assignments.
    :param k: the cluster index.
    :return: a new point which is the mean of the cluster based on the RGB values of each point.
    """
    cluster_points = dataset[cluster == k]

    mean_red = cluster_points['Red'].mean()
    mean_green = cluster_points['Green'].mean()
    mean_blue = cluster_points['Blue'].mean()

    return [mean_red, mean_green, mean_blue]


def k_means(rgb_mat):
    """
    the main function that we call to run the KMC algorithm
    :param rgb_mat: the usual matrix
    :return: the absolute best clusters in terms of MSE in a specified number of KMC runs
    """
    best_clusters = None
    mse_min = float('inf')  # start from infinity and obviously it'll update to a finite amount

    # run K-means multiple times to find the best clustering
    for i in range(300):  # TODO: why range 10? why not range 100?
        mse = 0
        # Initialize centroids
        centroids = k_means_ready(rgb_mat)
        # Perform clustering and update centroids
        clusters, distance_arr = split_and_update_clusters(rgb_mat, centroids)

        # Calculate the total variance
        for distance in distance_arr:
            mse += distance ** 2

        # Check if the current run has the lowest variance
        if mse < mse_min:
            mse_min = mse
            best_clusters = clusters

    return best_clusters
