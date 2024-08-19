import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def error_plot(data, condition, values):
    data_plot = data.copy()
    data_plot = data_plot[[values, condition]]
    data_plot = data_plot.groupby([condition]) \
        .agg(['mean', 'std']) \
        .reset_index()
    
    plt.errorbar(data_plot[condition], data_plot[values]['mean'],
                 yerr = data_plot[values]['std'],
                 fmt = 'o', capsize = 4)
    
    plt.show()