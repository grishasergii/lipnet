import sys
import getopt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
import os
import itertools
import numpy as np
from scipy import stats
    
import confusion_matrix as cf

def diff(x):
    a = np.array(x)
    return a[1:, 0] - a[0:-1, 0]

do_histograms = False
do_scatterplots = False
remove_previous = True

problem = 'lamellarity'
dir = '/home/sergii/Documents/microscopic_data/{}/'
path_to_json = dir + 'particles_repaired.json'
path_to_img = dir + 'images/without_padding/'

# set plot styles
matplotlib.style.use('ggplot')

# create a dataset
df = pd.read_json(path_to_json.format(problem))

# replace class int with names
df['Class'] = df['Class'].replace(to_replace=[3, 4, 5, 7, 8, 10],
                                  value=['Unilamellar', 'Multilamellar', 'Uncertain', 'Empty', 'Full', 'Uncertain'])

labels = df.Class.unique()
colors = {
    'Unilamellar': 'LightBlue',
    'Multilamellar': 'DarkGreen',
    'Uncertain': 'Red',
    'Empty': 'DarkGreen',
    'Full': 'LightBlue'
}

# split moments into separate columns
df[['M20', 'M02', 'M30', 'M03']] = df['Moments'].apply(lambda x: pd.Series(x))

#df.plot.scatter(x='Length', y='MembraneThickness')
#scatter_matrix(df[['Circularity', 'Area', 'Perimeter', 'Length', 'DiametersInPixels', 'MaximumWidth']], alpha=0.2, figsize=(6, 6), diagonal='kde')

columns = ['Area', 'Circularity', 'Perimeter', 'Length', 'MaximumWidth',
           'SignalToNoise', 'M20', 'M02', 'M30', 'M03']

output_path = 'output/figures/{}/{}/'

lower_percentile = 0.01
upper_percentile = 0.99

# histograms
if do_histograms:
    output_path_histogram = output_path.format(problem, 'histogram')
    if not os.path.isdir(output_path_histogram):
        os.makedirs(output_path_histogram)

    nbins = 50

    for c in columns:
        quantile = df[c].quantile([lower_percentile, upper_percentile]).values
        print '{} min: {} max: {}'.format(c, df[c].min(), df[c].max())
        print '{} lower: {} upper: {}'.format(c, quantile[0], quantile[1])
        ax = df[c].hist(bins=nbins, range=quantile)
        fig = ax.get_figure()
        fig.suptitle('{} {} histogram'.format(problem, c), fontsize=20)
        fig.savefig(output_path_histogram + 'histogram_{}.png'.format(c))
        plt.close(fig)

if do_scatterplots:
    output_path_scatterplots = output_path.format(problem, 'scatter_plot')
    if not os.path.isdir(output_path_scatterplots):
        os.makedirs(output_path_scatterplots)
    elif remove_previous:
        filelist = [f for f in os.listdir(output_path_scatterplots)]
        for f in filelist:
            os.remove(os.path.join(output_path_scatterplots, f))

    column_combinations = itertools.combinations(columns, 2)
    for c in column_combinations:
        print 'scatter plot: {} / {}'.format(c[0], c[1])
        ax = None
        quantile_x = df[c[0]].quantile([lower_percentile, upper_percentile]).values
        quantile_y = df[c[1]].quantile([lower_percentile, upper_percentile]).values
        for label in labels:
            if ax is None:
                ax = df.loc[df.Class == label].plot.scatter(x=c[0], y=c[1], color=colors[label], label=label,
                                                            xlim=quantile_x, ylim=quantile_y)
            else:
                ax = df.loc[df.Class == label].plot.scatter(x=c[0], y=c[1], color=colors[label], label=label, ax=ax,
                                                            xlim=quantile_x, ylim=quantile_y)

        fig = ax.get_figure()
        # fig.suptitle('{} scatter plot {} / {}'.format(problem, c[0], c[1]), fontsize=20)
        fig.savefig(output_path_scatterplots + 'scatter_plot_{}_{}.png'.format(c[0], c[1]))
        plt.close(fig)
    pass

l = df['RadialDensityProfile'].apply(len)
quantile = l.quantile([lower_percentile, upper_percentile]).values
print 'Radial density profile length min: {} max: {} mode: {}'.format(l.min(), l.max(), stats.mode(l.values))
print quantile

l = df['EdgeDensityProfile'].apply(len)
quantile = l.quantile([lower_percentile, upper_percentile]).values
print 'Edge density profile length min: {} max: {} mode: {}'.format(l.min(), l.max(), stats.mode(l.values))
print quantile

pass




#if __name__ == '__main__':
#    main()