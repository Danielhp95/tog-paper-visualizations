import pickle
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


def plot_basis_scatter(embedding, ax):
    # A hack because we now that each embedding consists of
    # 3 clusters of 1000 points each. These clusters are also
    # ordered (1st cluster: 0:999, 2nd: 1000:2000, 3rd: 2000:3000)
    xx = embedding[:, 0]
    yy = embedding[:, 1]
    num_basis = 3  # Rock, paper, scissors
    colors = cm.Spectral(np.linspace(0, 1, num_basis))

    basis_borders = [0, 1000, 2000]
    xx_basis_1 = xx[:basis_borders[1]]
    yy_basis_1 = yy[:basis_borders[1]]

    xx_basis_2 = xx[basis_borders[1]:basis_borders[2]]
    yy_basis_2 = yy[basis_borders[1]:basis_borders[2]]

    xx_basis_3 = xx[basis_borders[2]:]
    yy_basis_3 = yy[basis_borders[2]:]

    ax.scatter(xx_basis_1, yy_basis_1, color=colors[0])
    ax.scatter(xx_basis_2, yy_basis_2, color=colors[1])
    ax.scatter(xx_basis_3, yy_basis_3, color=colors[2])


def divide_points(points, divisions):
    '''
    Divides :param: points into sub np.arrays
    :param points: np.array of points
    :param divisions: int, number of sub np.arrays we wish to divide the original :param: points
    '''
    return np.array_split(points, divisions)


def link_points(points, ax, colour_map, alpha):
    '''
    Plots :param: points as points linked by lines on :param: ax Axes.
    The colour of the points and the lines is a 'smooth' colour transition from
    both colour extremes of the :param: colour_map
    '''
    number_of_points = points.shape[0]
    colours = [colour_map(1.*i/(number_of_points-1)) for i in range(number_of_points-1)]
    point_pairs = zip(points, points[1:]) # Creates a pairs of points (p_0, p_1), (p_1, p_2)...
    for point_pair, colour in zip(point_pairs, colours):
        numpy_point_pair = np.array(point_pair)
        ax.plot(numpy_point_pair[:, 0], numpy_point_pair[:, 1], '-o', color=colour, alpha=alpha)


def draw_arrow(ax, point_a, point_b, alpha, color):
    # Arrow params
    shape='full'
    head_starts_at_zero=True
    arrow_h_offset = 0.1  # data coordinates, empirically determined
    max_arrow_width=0.3125
    max_arrow_length = 1 - 1.2 * arrow_h_offset
    max_head_width = 4.5 * max_arrow_width
    max_head_length = 5 * max_arrow_width
    arrow_params = {'length_includes_head': True, 'shape': shape,
                    'head_starts_at_zero': head_starts_at_zero}
    # set the length of the arrow
    length = max_arrow_length
    width = max_arrow_width
    head_width = max_head_width
    head_length = max_head_length

    point_a = np.reshape( point_a, (-1))
    point_b = np.reshape( point_b, (-1))
    delta = point_b-point_a
    dx, dy = delta[0], delta[1]
    x_a, y_a = point_a[0], point_a[1]
    ax.arrow(x_a, y_a, dx, dy,
              fc=color, ec=color, alpha=alpha, width=width,
              head_width=head_width, head_length=head_length,
              **arrow_params
             )

def point_at_points(points, ax, colour_map, alpha):
    '''
    Plots :param: points as points linked by lines on :param: ax Axes.
    The colour of the points and the lines is a 'smooth' colour transition from
    both colour extremes of the :param: colour_map
    '''
    number_of_points = points.shape[0]
    colours = [colour_map(1.*i/(number_of_points-1)) for i in range(number_of_points-1)]
    point_pairs = zip(points, points[1:]) # Creates a pairs of points (p_0, p_1), (p_1, p_2)...
    for idx, (point_pair, colour) in enumerate(zip(point_pairs, colours)):
        numpy_point_pair = np.array(point_pair)
        if idx==0: ax.plot(numpy_point_pair[0, 0], numpy_point_pair[0, 1], '-o', color=colour, alpha=alpha)
        draw_arrow(ax, numpy_point_pair[0, :], numpy_point_pair[1, :], alpha=alpha, color=colour)


def plot_trajectory_evolution_in_embedded_space_with_arrows(points, divisions, ax, alpha=0.8):
    # Rainbow colour map:
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.1, 0.5, 0.5),
                     (0.2, 0.0, 0.0),
                     (0.4, 0.2, 0.2),
                     (0.6, 0.0, 0.0),
                     (0.8, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
            'green':((0.0, 0.0, 0.0),
                     (0.1, 0.0, 0.0),
                     (0.2, 0.0, 0.0),
                     (0.4, 1.0, 1.0),
                     (0.6, 1.0, 1.0),
                     (0.8, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
            'blue': ((0.0, 0.0, 0.0),
                     (0.1, 0.5, 0.5),
                     (0.2, 1.0, 1.0),
                     (0.4, 1.0, 1.0),
                     (0.6, 0.0, 0.0),
                     (0.8, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
    colour_map = matplotlib.colors.LinearSegmentedColormap('rainbow_colormap',cdict,256)

    divided_points = divide_points(points, divisions=divisions)
    centroids = np.array([np.median(sub_points, axis=0) for sub_points in divided_points])
    point_at_points(centroids, ax, colour_map, alpha)



######################## Main 3 functions
def plot_basis_only(embedding, ax):
    basis_embedding = embedding[:3000]  # HACK: we know the first 3000 points are the basis
    plot_basis_scatter(basis_embedding, ax)
    return ax


def plot_basis_and_first_episodes(embedding, ax):
    plot_basis_scatter(embedding[:3000], ax)
    first_500_eps = embedding[3000:3500]
    ax.scatter(first_500_eps[:, 0], first_500_eps[:, 1], color='green')
    s = np.linspace(0, 3, 10)[5]
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
    sns.kdeplot(first_500_eps[:, 0], first_500_eps[:, 1],
                cmap=cmap, fill=True, cut=1, ax=ax, alpha=0.8)


def plot_basis_first_episodes_and_trajectory_evolution(embedding, ax):
    #plot_basis_and_first_episodes(embedding, ax)
    plot_basis_only(embedding, ax)
    plot_trajectory_evolution_in_embedded_space_with_arrows(
        points=embedding[3000:5000],
        divisions=4,
        ax=ax
    )
########################


def main(embedding, save):
    embedding = pickle.load(open(args.embedding, 'rb'))

    # Create matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax = plot_basis_only(embedding, ax)

    if save:
        plt.savefig('rirrps_basis_scatter_plot.pdf', format='pdf', dpi=1000, transparent=True)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    plot_basis_and_first_episodes(embedding, ax)

    if args.save:
        plt.savefig('rirrps_basis_and_first_episodes_plot.pdf', format='pdf', dpi=1000, transparent=True)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    plot_basis_first_episodes_and_trajectory_evolution(embedding, ax)

    if args.save:
        plt.savefig('rirrps_basis_first_episodes_and_trajectory_evolution_plot.pdf', format='pdf', dpi=1000, transparent=True)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='Path to embedding file')
    parser.add_argument('--save', action='store_true', help='Whether to save embedding in embedding_plot.eps')
                        # defaults to false
    args = parser.parse_args()
    main(args.embedding, args.save)
