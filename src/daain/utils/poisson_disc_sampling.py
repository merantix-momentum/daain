from itertools import product
from typing import List, Union

import numpy as np
from scipy.spatial.distance import cdist, euclidean

# def _uniform_sampling_annulus_cdf(smaller_radius, larger_radius, dims=2, radial_coordinates=False):
#    """Random annulus sampling using math. Much faster than rejection sampling."""
#    theta = np.random.uniform(0, 2 * np.pi, size=dims - 1)
#    # normalising_constant = 2 / (larger_radius ** 2 - smaller_radius ** 2)
#    # r = np.sqrt(2 * np.random.uniform(0, 1) / normalising_constant + smaller_radius ** 2)
#
#    u = np.random.uniform()
#    r = np.power(u * larger_radius + (1 - u) * smaller_radius, 1 / dims)
#
#    if radial_coordinates:
#        return theta, r
#    else:
#        if dims == 2:
#            return r * np.cos(theta), r * np.sin(theta)
#        elif dims == 3:
#            return (r * np.sin(theta[0]) * np.cos(theta[1]),
#                    r * np.sin(theta[0]) * np.sin(theta[1]),
#                    r * np.cos(theta[0]))
#        else:
#            raise NotImplementedError(
#                "Cartesian coordinates transformation not implemented for more than 3 dimensions.")


def _uniform_sampling_annulus_rejection_sampling(
    smaller_radius: Union[float, List[float]], larger_radius: Union[float, List[float]], dims: int = 2
) -> List[float]:
    """Simple method that can generate n-dimensional points in the n-dimensional annulus defined by the two radii.

    This uses rejection sampling, so be aware that it will take some time. Not the best method - IMO,
    check out _uniform_sampling_annulus_cdf for a smarter and much faster way. This however is guaranteed to be uniform.
    """
    # return first value that fits inside
    while True:
        p = np.random.uniform(-larger_radius, larger_radius, size=dims)
        if smaller_radius <= np.linalg.norm(p, ord=2) <= larger_radius:
            return p


def _grid_coords(point, cellsize):
    return tuple(np.floor(point / cellsize).astype(int))


def poisson_disc_samples(
    shapes: List[int],
    minimum_distance_between_samples: float,
    samples_chosen_for_rejection: int = 30,
    distance=euclidean,
):
    """Returns the indices of the samples using "Fast Poisson Disk Sampling in Arbitrary Dimensions", aka blue noise
    from Robert Bridson https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf

    However there is a problem with 'picking points uniformly in an n-dimensional annulus' in the original paper.

    The simple option is to use rejection sampling: Use a n-dimensional square, draw the coordinates uniformly for each
    dimension and reject numbers that don't fall into the annulus. This is however super inefficient if the smaller
    radius is large.

    https://math.stackexchange.com/questions/1885630/random-multivariate-in-hyperannulus
    So I use the following (source: https://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus):

    - draw theta uniformly from [0, 2 * pi]
    - draw r from the power-law distribution r^1

    Args:
        shapes: extent of the sample domain
        minimum_distance_between_samples: , samples_chosen_for_rejection=30, distance=euclidean


    """

    def fits(p, grid_, p_coords_, shapes_):
        # checking if there is an element in the surrounding grid points
        neighborhoods = product(
            *[list(range(max(c - 2, 0), min(c + 3, grid_.shape[i]))) for i, c in enumerate(p_coords_)]
        )
        # print(f"grid {grid_.shape}")
        # print(p_coords_)
        # print(list(neighborhoods))
        for n in neighborhoods:
            g = grid_[n]
            # this could be optimised with using only the squared min distance
            if g is not None and distance(p, g) <= minimum_distance_between_samples:
                return False

        # if p is out of boundary
        if np.any(p < 0) or any(p[i] > s for i, s in enumerate(shapes_)):
            # print(f"rejected because of boundary: {p}")
            return False

        return True

    # step 0, initializing the n-dimensional background grid
    cellsize = minimum_distance_between_samples / np.sqrt(len(shapes))

    background_grid = np.zeros(np.ceil(np.array(shapes) / cellsize).astype(int), dtype=object)
    background_grid[:] = None  # no sample, non-negative integer gives the index of the sample located in a cell

    # step 1, select initial sample, x_0
    initial_sample = np.random.random(size=len(shapes)) * np.array(shapes)
    idx = _grid_coords(initial_sample, cellsize)
    background_grid[idx] = initial_sample
    active_list = [idx]

    # step 2, while active_list is not empty, pick a random item from it
    run_idx = 0
    while active_list:
        # `i` in original algorithm
        idx_chosen_point = np.random.randint(len(active_list))

        chosen_point = active_list[idx_chosen_point]
        # print(f"curr cp: {chosen_point}")

        # generate k new points
        # uniformly in the annulus around the chosen point
        new_points = [
            np.array(
                _uniform_sampling_annulus_rejection_sampling(
                    minimum_distance_between_samples, minimum_distance_between_samples * 2, dims=len(shapes)
                )
            )
            + background_grid[chosen_point]
            for _ in range(samples_chosen_for_rejection)
        ]

        # do_plot = any((run_idx < 4,
        #               (run_idx < 100 and run_idx % 20 == 0),
        #               (run_idx < 1000 and run_idx % 100 == 0),
        #               run_idx % 1000 == 0))
        # do_plot = False
        # if do_plot:
        #    fig = plt.figure(figsize=(10, 10))
        #    ax = fig.add_subplot(111, projection='3d')

        #    x, y, z = list(zip(*new_points))
        #    ax.scatter(x, y, z, c='r', label='new gen')

        #    x, y, z = list(zip(*np.array([p for p in background_grid.reshape(-1).tolist() if p is not None])))
        #    ax.scatter(x, y, z, c='g', label='old points')

        #    ax.set_xlim(0, shapes[0])
        #    ax.set_ylim(0, shapes[1])
        #    ax.set_zlim(0, shapes[2])

        new_points = [p for p in new_points if fits(p, background_grid, _grid_coords(p, cellsize), shapes)]

        if len(new_points):
            for p in new_points:
                if fits(p, background_grid, _grid_coords(p, cellsize), shapes):
                    # TODO detect if new point is out of boundary -> discard them
                    p_coords = _grid_coords(p, cellsize)
                    background_grid[p_coords] = p
                    active_list.append(p_coords)

            # if do_plot:
            #    x, y, z = list(zip(*new_points))
            #    ax.scatter(x, y, z, c='b', label='new gen after filter')
        else:
            active_list[idx_chosen_point] = active_list[-1]
            active_list.pop()

        # if do_plot:
        #    ax.scatter(initial_sample[0], initial_sample[1], initial_sample[2], label='origin')
        #    ax.scatter(background_grid[chosen_point][0], background_grid[chosen_point][1],
        #               background_grid[chosen_point][2], label='step origin')
        #    plt.title(f"idx: {run_idx}, n news: {len(new_points)}, total {len(x)}")
        #    plt.legend()
        #    plt.show()

        run_idx += 1

    return np.array([p for p in background_grid.reshape(-1).tolist() if p is not None])


def _get_lattice_shape(original_shape, n_samples=None):
    old_volumne = np.prod(original_shape)
    scale_factor = np.power(n_samples / old_volumne, 1 / len(original_shape))
    s = np.array(original_shape) * scale_factor
    return np.round(s).astype(int)


def _get_initial_weights(original_shape, n_samples, perturbation=0):
    dim = len(original_shape)
    lattice_shape = _get_lattice_shape(original_shape, n_samples=n_samples)

    # initlize the weights according to their positions in space
    # remember how reshaping and product works in numpy
    # here is some example code to see it: `np.array(list(product(range(4), range(3)))).reshape(4, 3, 2)`
    # it will produce a grid with width / height of 4 / 3 and two dimensions (since it's a 2 dim grid)
    # this is not the fastest method! using meshgrid and some funky numpy would be better, but this is somewhat clean
    grid_spaces = [
        np.linspace(0, original_shape[i], lattice_shape[i], endpoint=False, dtype=float)
        + (original_shape[i] / (2 * lattice_shape[i]))
        for i in range(dim)
    ]
    weights = np.array(list(product(*grid_spaces))).reshape((*lattice_shape, dim))
    weights += np.random.rand(*weights.shape) * perturbation  # add some small noise to it

    return weights


def gridify_poisson_disc_sampling(samples, original_shape, min_distance):
    initial_weights = _get_initial_weights(original_shape, samples.shape[0])
    init_weights = initial_weights.reshape(-1, len(original_shape))

    # find closest pair between weights and samples
    weights = init_weights.copy()
    dist_pairs = cdist(weights, samples)

    # since we have less weights than samples
    for _ in range(weights.shape[0]):
        closest_pair_idx = np.unravel_index(np.argmin(dist_pairs), dist_pairs.shape)  # (idx for weight, idx for sample)

        if dist_pairs[closest_pair_idx] < min_distance:
            weights[closest_pair_idx[0], :] = samples[closest_pair_idx[1], :]

            # remove the weight and sample from the pool
            dist_pairs[:, closest_pair_idx[1]] = np.finfo(np.float).max
            dist_pairs[closest_pair_idx[0], :] = np.finfo(np.float).max

    return weights.reshape(initial_weights.shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def _plot(samples):
        x, y = list(zip(*samples))
        plt.plot(x, y, "o")
        plt.show()

    def _splitter(samples, pred):
        ms, ns = [], []

        for s in samples:
            (ms if pred(s) else ns).append(s)

        return ms, ns

    # grid = poisson_disc_samples([10, 10], 2)
    # _plot(grid)

    # _plot(poisson_disc_samples([10, 10], 1))
    # _plot(poisson_disc_samples([1024, 512], 10))

    # for i in range(len(grid)):
    #    print(any(euclidean(grid[i], grid[j]) < 2 for j in range(len(grid)) if j != i))

    # smaller_radius = 0.3
    # larger_radius = 1

    # for fn in [_uniform_sampling_annulus_rejection_sampling]:
    #    try:
    #        annulus_sampled = [fn(smaller_radius, larger_radius) for _ in range(100000)]
    #        # split radius to give a ratio of ~1
    #        split_radius = np.sqrt((larger_radius ** 2 + smaller_radius ** 2) / 2)
    #        inner, outer = _splitter(annulus_sampled, lambda x: np.linalg.norm(x, ord=2) <= split_radius)
    #        np.testing.assert_allclose(len(inner) / len(outer), 1, rtol=1e-03)
    #    except AssertionError as e:
    #        print(f"error with (not close enough) {fn}, {e}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    width, height, activations = 10, 10, 100

    grid = poisson_disc_samples([width, height, activations], 1)
    x, y, z = list(zip(*grid))
    ax.scatter(x, y, z)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, activations)

    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_zlabel("activations")
    plt.show()
