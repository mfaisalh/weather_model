import matplotlib.pyplot as plt

import numpy as np



def visualize_data(data, title, output_file):

    """Visualizes the weather data and saves it to a file."""

    plt.figure(figsize=(10, 5))

    plt.imshow(data, cmap='coolwarm', interpolation='nearest')

    plt.colorbar()

    plt.title(title)

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')

    plt.savefig(output_file)

    plt.close()

