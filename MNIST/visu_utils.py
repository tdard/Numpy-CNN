import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(costs, display=True, figure=None):
    iterations = np.arange(1, costs.size+1)
    # Initialize new figure if needed
    fig = figure if figure is not None else plt.figure()
    # Get current axis and scatter
    axes = fig.gca() 
    axes.scatter(iterations, costs)
    if display:
        plt.show()
    return fig