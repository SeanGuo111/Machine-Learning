from matplotlib import pyplot as plt
import numpy as np

def plot(x_points, y_points, intercept, slope):
    """Plot a line and points from slope and intercept"""
    # Plot points
    plt.scatter(x_points, y_points)

    # Plot line
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    plt.show()

# For your model:
# plot(xData, yData, W0value, W1value)