from flash1dkmeans import kmeans_1d


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

centroids, cluster_borders = kmeans_1d(data, n_clusters=3)

print(centroids)
print(cluster_borders)
