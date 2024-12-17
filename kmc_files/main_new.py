import matplotlib.pyplot as plt
from RGB_sort import get_rgb_dominance
from kmc_alg import k_means
image_directory_1 = "fruits"
rgb_df = get_rgb_dominance(image_directory_1)

# Specify the number of clusters
num_clusters = 3

# Prepare the data for clustering (only RGB columns)
# kmeans = KMeans(n_clusters=num_clusters, n_init=10)
rgb_df['Cluster'] = k_means(rgb_df)

# Display the DataFrame with the cluster information
print("\nRGB Values of Images with Clusters:")
print(rgb_df)

# Step 4: Plot RGB Dominance in 3D Space with Clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract R, G, B values, cluster labels, and image names
red_values = rgb_df['Red']
green_values = rgb_df['Green']
blue_values = rgb_df['Blue']
clusters = rgb_df['Cluster']
image_names = rgb_df['Image Name']

# Use different colors for each cluster
colors = plt.cm.get_cmap("viridis", num_clusters)

# Scatter plot with clusters
scatter = ax.scatter(red_values, green_values, blue_values,
                     c=clusters, cmap=colors, s=100, edgecolors='k')

# Set plot labels
ax.set_title('3D Visualization of RGB Dominance with K-means Clusters')
ax.set_xlabel('Red Dominance')
ax.set_ylabel('Green Dominance')
ax.set_zlabel('Blue Dominance')

# Display the color bar for clusters
plt.colorbar(scatter, ax=ax, label='Cluster')

# Show the plot
plt.show()
