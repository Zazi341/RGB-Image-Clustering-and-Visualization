import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Function to Extract RGB Dominance from Images
def get_rgb_dominance(image_directory):
    # Initialize empty lists to store the RGB dominance of each image
    image_name = []
    red_dominance = []
    green_dominance = []
    blue_dominance = []

    # Loop through each file in the given directory
    for filename in os.listdir(image_directory):
        file_path = os.path.join(image_directory, filename)
        # Open the image using PIL
        with Image.open(file_path) as img:
            # Convert image to RGB mode if not already in that mode
            img = img.convert('RGB')

            # Convert the image into a NumPy array
            img_array = np.array(img)

            # Calculate the average RGB values across all pixels
            avg_red = np.mean(img_array[:, :, 0])
            avg_green = np.mean(img_array[:, :, 1])
            avg_blue = np.mean(img_array[:, :, 2])

            # Append the average RGB values to the respective lists
            red_dominance.append(avg_red)
            green_dominance.append(avg_green)
            blue_dominance.append(avg_blue)
            image_name.append(filename)

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Image Name': image_name,
        'Red': red_dominance,
        'Green': green_dominance,
        'Blue': blue_dominance
    })

    return df


# Extract RGB Dominance from Images in Directory
image_directory_1 = "fruits"  # Replace with your image directory path
rgb_df = get_rgb_dominance(image_directory_1)

# Display the DataFrame as a table
print("\nRGB Values of Images:")
print(rgb_df)

# Step 3: Plot RGB Dominance in 3D Space
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract R, G, B values and image names
red_values = rgb_df['Red']
green_values = rgb_df['Green']
blue_values = rgb_df['Blue']
image_names = rgb_df['Image Name']

# Scatter plot for RGB values
scatter = ax.scatter(red_values, green_values, blue_values,
                     c=np.array([red_values, green_values, blue_values]).T / 255.0,
                     s=100, edgecolors='k')

# Add labels to each point

# for i, name in enumerate(image_names):
# ax.text(red_values[i], green_values[i], blue_values[i], name, fontsize=8)

# Set plot labels
ax.set_title('3D Visualization of RGB Dominance')
ax.set_xlabel('Red Dominance')
ax.set_ylabel('Green Dominance')
ax.set_zlabel('Blue Dominance')

# Display the plot
plt.show()
