from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def main():
    problem = 'lamellarity'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + 'particles_repaired.json'
    output_path = os.path.join('output/figures/', problem)

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

    columns = ['Area', 'Circularity', 'Perimeter', 'Length', 'MaximumWidth',
               'SignalToNoise', 'M20', 'M02', 'M30', 'M03']

    x = df[columns].values.copy()
    y = df.Class.values.copy()

    pca = PCA(n_components=2)
    x_r = pca.fit(x).transform(x)
    xlim = (np.percentile(x_r[:, 0], 0.5),
            np.percentile(x_r[:, 0], 99.5))
    ylim = (np.percentile(x_r[:, 1], 0.5),
            np.percentile(x_r[:, 1], 99.5))
    print 'explained variance ration (first two components): {}'.format(str(pca.explained_variance_ratio_))

    # set plot styles
    #matplotlib.style.use('ggplot')

    fig = plt.figure()
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    for i, target_name in enumerate(labels):
        plt.scatter(x_r[y == target_name, 0],
                    x_r[y == target_name, 1],
                    c=colors[target_name],
                    label=target_name,
                    lw=0)
    plt.legend()
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    fig.savefig(os.path.join(output_path, 'pca.png'), bbox_inches='tight')
    pass


if __name__ == '__main__':
    main()