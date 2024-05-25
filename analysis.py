import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import symnmf

EPSILON = 1e-4  # Defines the precision for determining the convergence of centroids.
ITER_NUM = 300  # Default number of iterations if none is specified by the user.
GENERAL_ERROR = "An Error Has Occurred"


# This function performs the K-means clustering algorithm on a given set of data.
# k: the number of clusters to form.
# iter: the maximum number of iterations to perform.
# input_data: the path to the input data file.
def find_clusters(k, data_points):
    # Initialize centroids using the first k data points.
    centroids = data_points[:k]
    # Initialize an empty cluster list for each centroid.
    clusters = [[] for _ in range(k)]

    # Iterate over the maximum number of iterations.
    for iteration_number in range(ITER_NUM):
        epsilon_counter = 0  # Counter to track how many centroids have converged.

        # For each data point, find the closest centroid and assign the data point to that cluster.
        for data_point in data_points:
            i = get_closest_centroid_index(centroids, data_point)
            clusters[i].append(data_point)

        # For each cluster, calculate the mean of the points (the new centroid) and update its position.
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                # new_centroid = [round(sum(dim) / len(cluster), 4) for dim in zip(*cluster)]
                new_centroid = [round(dim / len(cluster), 4) for dim in map(sum, zip(*cluster))]
                if calculate_distance(centroids[i], new_centroid) < EPSILON:
                    epsilon_counter += 1
                centroids[i] = new_centroid
            else:
                epsilon_counter += 1
        # If this is not the last iteration, reset the clusters for the next round.
        if iteration_number < ITER_NUM - 1:
            res = clusters
            clusters = [[] for _ in range(k)]

        # If all centroids have converged, stop the iteration and return the final centroid positions.
        if epsilon_counter == k:
            return res

    # If reached the maximum number of iterations without all centroids converging, return current centroids.
    return res


def calculate_distance(vector1, vector2):
    sum_squares = 0
    for i in range(len(vector1)):
        sum_squares += (vector1[i] - vector2[i]) ** 2
    distance = sum_squares ** 0.5
    return distance


# This function finds the index of the closest centroid to a given data point.
# It calculates the euclidean distance from the data point to each centroid and returns the index of the closest.
def get_closest_centroid_index(centroids, data_point):
    distances = [calculate_distance(data_point, centroid) for centroid in centroids]
    return distances.index(min(distances))


def construct_cluster_array(clusters, data_points):
    cluster_arr = [0] * len(data_points)
    for i, data_point in enumerate(data_points):
        break_out = False
        for j, cluster in enumerate(clusters):
            for vector in cluster:
                if data_point == vector:
                    cluster_arr[i] = j
                    break_out = True
                    break
            if break_out:
                break
    return cluster_arr


def main():
    if len(sys.argv) != 3:
        print(GENERAL_ERROR)
        quit()

    k = int(sys.argv[1])
    file_name = sys.argv[2]
    file_df = pd.read_csv(file_name, header=None)
    if k <= 1 or k >= file_df.shape[0]:
        print(GENERAL_ERROR)
        quit()

    dim = file_df.shape[1]
    data_points = file_df.values.tolist()

    n = file_df.shape[0]
    # this creates H (the decomposition matrix)
    try:
        sym_mat = symnmf.get_symnmf(n, k, dim, data_points)
        # this creates an array, such that the ith element is the number of the cluster that the ith vector belongs to
        sym_clusters = np.argmax(sym_mat, axis=1)
        kmeans_clusters = construct_cluster_array(find_clusters(k, data_points), data_points)
        # this prints the silhouette score of symnmf
        print("nmf: {:.4f}".format(silhouette_score(data_points, sym_clusters)))
        print("kmeans: {:.4f}".format(silhouette_score(data_points, kmeans_clusters)))
    except RuntimeError:
        print(GENERAL_ERROR)
        quit()

# Run the main function if this script is run as the main module.
if __name__ == "__main__":
    main()
