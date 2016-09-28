from dataset.dataset_images import DatasetImages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import helpers
from matplotlib.colors import LinearSegmentedColormap


path_to_json = '/home/sergii/Documents/microscopic_data/{}/particles_repaired.json'
path_to_img_with_padding = '/home/sergii/Documents/microscopic_data/{}/images/particles/'
path_to_img_without_padding = '/home/sergii/Documents/microscopic_data/{}/images/without_padding/'


def main(problem_name):

    dataset = DatasetImages.from_json(path_to_json.format(problem_name),
                                      path_to_img_without_padding.format(problem_name),
                                      img_size=None)
    df = dataset.get_image_size_stats()
    if problem_name == 'packiging':
        df['Class'] = df['Class'].replace(to_replace=[0, 1, 2],
                        value=['Empty', 'Full', 'Uncertain'])
        labels = ['Full', 'Empty', 'Uncertain']
    else:
        df['Class'] = df['Class'].replace(to_replace=[0, 1, 2],
                        value=['Unilamellar', 'Multilamellar', 'Uncertain'])
        labels = ['Unilamellar', 'Multilamellar', 'Uncertain']



    out_path = 'output/figures/{}/image_size/'.format(problem_name)
    helpers.prepare_dir(out_path, empty=True)

    lower_percentile = 0
    upper_percentile = 0.999
    colors = {
        'Unilamellar': 'LightBlue',
        'Multilamellar': 'DarkGreen',
        'Uncertain': 'Red',
        'Empty': 'DarkGreen',
        'Full': 'LightBlue'
    }

    ax = None
    quantile_x = df['Width'].quantile([lower_percentile, upper_percentile]).values
    quantile_y = df['Height'].quantile([lower_percentile, upper_percentile]).values
    for label in labels:
        print label
        if ax is None:
            ax = df.loc[df.Class == label].plot.scatter(x='Width', y='Height', color=colors[label], label=label,
                                                        xlim=quantile_x, ylim=quantile_y)
        else:
            ax = df.loc[df.Class == label].plot.scatter(x='Width', y='Height', color=colors[label], label=label, ax=ax,
                                                        xlim=quantile_x, ylim=quantile_y)

    fig = ax.get_figure()
    fig.savefig(out_path + 'scatter_plot_{}_{}.png'.format('width', 'height'),
                bbox_inches='tight')
    plt.close(fig)

    print 'heatmaps'
    # make heatmaps
    x_bins = np.linspace(quantile_x[0], quantile_x[1], num=50)
    y_bins = np.linspace(quantile_y[0], quantile_y[1], num=50)
    for label in labels:
        print label
        x = df.loc[df.Class == label].Width.values
        y = df.loc[df.Class == label].Height.values
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins))

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        vmax = np.amax(heatmap)
        cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'white'),
                                                            (0.00000000000001, '#eeeeee'),
                                                            (1, 'red')])
        plt.clf()
        plt.imshow(heatmap,
                   extent=extent,
                   interpolation='nearest',
                   origin='lower',
                   cmap=cmap)

        plt.colorbar()
        plt.savefig(out_path + 'heatmap_{}.png'.format(label.lower()),
                    bbox_inches='tight')
    pass

if __name__ == '__main__':
    main('lamellarity')
    main('packiging')