# RGB Image Clustering and 3D Visualization

A Python-based project for clustering images by their RGB color dominance using K-Means, with interactive 3D visualization. This tool processes images, extracts RGB data, performs clustering, and visualizes the results in a 3D scatter plot.

## Features

### 1. RGB Extraction
- **Image Processing**: Processes images from a specified directory.
- **Color Analysis**: Extracts average Red, Green, and Blue (RGB) values for each image.
- **Data Structuring**: Stores the RGB values in a structured DataFrame.

### 2. K-Means Clustering
- **Custom Implementation**: Implements a custom K-Means clustering algorithm.
- **Centroid Initialization**: Uses the K-Means++ initialization method for better accuracy and faster convergence.
- **Optimization**: Iteratively updates clusters and centroids until convergence.
- **Image Assignment**: Assigns each image to one of the specified clusters (default: 3 clusters).

### 3. 3D Visualization
- **Scatter Plot**: Visualizes the RGB values of images in a 3D scatter plot.
- **Cluster Differentiation**: Assigns unique colors to each cluster for better distinction.
- **Interactivity**: Provides an interactive 3D visualization using Matplotlib.
- **Insights**: Displays clustering patterns based on RGB dominance.

## Usage
This project provides an intuitive way to analyze and visualize image clustering based on RGB values. It is ideal for tasks involving color-based image grouping, creative analysis, or visual data exploration.
