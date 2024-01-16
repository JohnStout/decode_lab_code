# plotter tools
import matplotlib.pyplot as plt
import numpy as np

def scat_data(y, x_label: str, y_label: str):
    '''
    Brain dead scatter plot
    
    '''

    x = np.linspace(0,len(y),len(y))
    plt.scatter(x, y, s=60, alpha=0.7, edgecolors="k")
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # fit a linear regression line to the data
    b, a = np.polyfit(x, y, deg=1)

    # Plot regression line
    plt.plot(x, a + b * x, color="r", lw=2.5)    