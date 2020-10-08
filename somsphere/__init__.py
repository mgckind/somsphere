"""
.. module:: SOMZ
.. moduleauthor:: Matias Carrasco Kind

"""
__author__ = 'Matias Carrasco Kind'

import os
import random

import core
import healpy as hp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy
from matplotlib import collections
from matplotlib.colors import colorConverter

from somsphere.models import Topology, DY, SomType
from somsphere.utils import compute_distance, get_sigma, get_alpha, is_power_2, timeit, count_modified_cells, \
    get_best_cell


class SOMap(object):
    """
    Create a som class instance

    :param float X: Attributes array (all columns used)
    :param float Y: Attribute to be predicted (not really needed, can be zeros)
    :param str topology: Which 2D topology, 'grid', 'hex' or 'sphere'
    :param str som_type: Which updating scheme to use 'online' or 'batch'
    :param int n_top: Size of map,  for grid Size=n_top*n_top,
        for hex Size=n_top*(n_top+1[2]) if n_top is even[odd] and for sphere
        Size=12*n_top*n_top and top must be power of 2
    :param  int n_iter: Number of iteration the entire sample is processed
    :param bool periodic: Use periodic boundary conditions, valid for 'hex' and 'grid' only
    :param dict dict_dim: dictionary with attributes names
    :param float alpha_start: Initial value of alpha
    :param float alpha_end: End value of alpha
    :param str importance: Path to the file with importance ranking for attributes, default is none
    """

    def __init__(self, X, Y, topology='grid', som_type='online', n_top=28, n_iter=30, alpha_start=0.8,
                 alpha_end=0.5, periodic=False, dict_dim=None, importance=None):
        if topology == 'sphere' and not is_power_2(n_top):
            raise Exception("n_top must be power of 2")

        self.X, self.Y = X, Y
        self.topology: Topology = Topology(topology)
        self.som_type: SomType = SomType(som_type)
        self.n_top = n_top
        self.n_iter = n_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.periodic = periodic
        self.dict_dim = dict_dim
        self.dist_lib, self.n_pix = compute_distance(self.topology, self.n_top, periodic=self.periodic)
        self.n_row, self.n_col = numpy.shape(X)
        self.weights = (numpy.random.rand(self.n_col, self.n_pix)) + self.X[0][0]
        importance = numpy.ones(self.n_col) if importance is None else importance
        self.importance = importance / numpy.sum(importance)

    def __update_weights(self, input_weights):
        self.weights = input_weights if input_weights is not None else self.weights

    @timeit
    def __create_map_online(self, random_order=True, eval_map=False):
        t, total_t = 0, self.n_iter * self.n_row
        sigma_0 = self.dist_lib.max()
        sigma_f = numpy.min(self.dist_lib[numpy.where(self.dist_lib > 0.)])

        for it in range(self.n_iter):
            alpha = get_alpha(self.alpha_end, self.alpha_start, t, total_t)
            sigma = get_sigma(sigma_f, sigma_0, t, total_t)
            random_indices = random.sample(range(self.n_row), self.n_row) if random_order else numpy.arange(
                self.n_row)
            self.weights = core.create_map_online(self.X, self.dist_lib, self.weights, random_indices, self.importance,
                                                  alpha, sigma, self.n_row, self.n_col, self.n_pix, 1)
            t += self.n_row

            if eval_map:
                self.evaluate_map()
                self.save_map(itn=it)

    @timeit
    def __create_map_batch(self, eval_map=False):
        t, total_t = 0, self.n_iter * self.n_row
        sigma_0 = self.dist_lib.max()
        sigma_f = numpy.min(self.dist_lib[numpy.where(self.dist_lib > 0.)])

        for it in range(self.n_iter):
            sigma = get_sigma(sigma_f, sigma_0, t, total_t)
            accum_w = numpy.zeros((self.n_col, self.n_pix))
            accum_n = numpy.zeros(self.n_pix)
            self.weights = core.create_map_batch(self.X, self.dist_lib, self.weights, self.importance, accum_w,
                                                 accum_n, sigma, self.n_row, self.n_col, self.n_pix, 1)
            t += self.n_row

            if eval_map:
                self.evaluate_map()
                self.save_map(itn=it)

    def create_map(self, input_weights=None, random_order=True, eval_map=False):
        """
        This functions actually create the maps, it uses
        random values to initialize the weights
        """
        self.__update_weights(input_weights)

        if self.som_type == SomType.ONLINE:
            self.__create_map_online(random_order=random_order, eval_map=eval_map)
        elif self.som_type == SomType.BATCH:
            self.__create_map_batch(eval_map=eval_map)
        else:
            raise Exception(f"Unknown type: {self.som_type}")

    def evaluate_map(self, input_x=None, input_y=None):
        """
        This functions evaluates the map created using the input Y or a new Y (array of labeled attributes)
        It uses the X array passed or new data X as well, the map doesn't change

        :param float input_x: Use this if another set of values for X is wanted using
            the weights already computed
        :param float input_y: One  dimensional array of the values to be assigned to each cell in the map
            based on the in-memory X passed
        """
        self.y_vals, self.i_vals = {}, {}
        in_x = self.X if input_x is None else input_x
        in_y = self.Y if input_y is None else input_y
        for i in range(len(in_x)):
            inputs = in_x[i]
            best, _ = core.get_best_cell(inputs, self.importance, self.weights, len(inputs), self.n_pix, 1)
            best = best[0]
            if best not in self.y_vals:
                self.y_vals[best] = []
            self.y_vals[best].append(in_y[i])
            if best not in self.i_vals:
                self.i_vals[best] = []
            self.i_vals[best].append(i)

    def predict(self, line, best=True):
        """
        Get the predictions  given a line search, where the line
        is a vector of attributes per individual object fot the
        10 closest cells if best set to False; otherwise return the
        BEST cell.

        :param float line: input data to look in the tree
        :param bool best: Set to True to get only the best cell; otherwise the 10 closest cells will be returned
        :return: array with the cell content
        """
        bests, _ = core.get_best_cell(line, self.importance, self.weights, len(line), self.n_pix, 10)
        if best:
            return bests[0]
        for ib in range(10):
            if bests[ib] in self.y_vals:
                return self.y_vals[bests[ib]]
        return numpy.array([-1.])

    def save_map(self, filename='SOM', path=None, itn=-1):
        """
        Saves the map and its dictionary format

        :param int itn: Number of map to be included on path, use -1 to ignore this number
        :param str filename: Name of output file
        :param str path: path for the output file
        """
        som = {'weights': self.weights, 'y_vals': self.y_vals, 'i_vals': self.i_vals, 'topology': self.topology,
               'n_top': self.n_top, 'n_pix': self.n_pix}

        path = os.getcwd() + '/' if path is None else path
        if not os.path.exists(path):
            os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            filename += ff

        numpy.save(path + filename, self)
        numpy.save(path + filename + ".txt", som)

    def plot_map(self, min_m=-100, max_m=-100, cbar=True):
        """
        Plots the map after evaluating, the cells are colored with the mean value inside each
        one of them

        :param float min_m: Lower limit for coloring the cells, -100 uses min value
        :param float max_m: Upper limit for coloring the cells, -100 uses max value
        :param bool cbar: Include a colorbar True/False
        """

        if self.topology == Topology.SPHERE:
            self.__plot_map_sphere(min_m, max_m, cbar)

        elif self.topology == Topology.GRID:
            self.__plot_map_grid(min_m, max_m, cbar)

        elif self.topology == Topology.HEX:
            self.__plot_map_hex(min_m, max_m, cbar)
        else:
            raise Exception(f"Unknown topology: {self.topology}")

    def __plot_map_sphere(self, min_m, max_m, cbar):
        M = numpy.zeros(self.n_pix) + hp.UNSEEN
        for i in range(self.n_pix):
            if i in self.y_vals:
                M[i] = numpy.mean(self.y_vals[i])
        plt.figure(10, figsize=(8, 8), dpi=100)
        min_m = M[numpy.where(M > -10)].min() if min_m == -100 else min_m
        max_m = M.max() if max_m == -100 else max_m
        hp.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=cbar)
        plt.show()

    def __plot_map_grid(self, min_m, max_m, cbar):
        M = numpy.zeros(self.n_pix) - 20.
        for i in range(self.n_pix):
            if i in self.y_vals:
                M[i] = numpy.mean(self.y_vals[i])
        M_new = numpy.reshape(M, (self.n_top, self.n_top))
        plt.figure(figsize=(8, 8), dpi=100)
        min_m = M_new[numpy.where(M_new > -10)].min() if min_m == -100 else min_m
        max_m = M_new.max() if max_m == -100 else max_m

        M_plot = plt.imshow(M_new, origin='center', interpolation='nearest', cmap=cm.jet, vmin=min_m, vmax=max_m)
        M_plot.cmap.set_under("grey")
        if cbar:
            plt.colorbar()
        plt.axis('off')

        plt.show()

    def __plot_map_hex(self, min_m, max_m, cbar):
        ptr = 0
        x_l, y_l = numpy.arange(0, self.n_top, 1.), numpy.arange(0, self.n_top, DY)
        nx, ny = len(x_l), len(y_l)
        n_pix = nx * ny
        b_x, b_y = numpy.zeros(n_pix), numpy.zeros(n_pix)
        for y_idx in range(ny):
            for x_idx in range(nx):
                b_x[ptr] = x_l[x_idx] + 0. if y_idx % 2 == 0 else 0.5
                b_y[ptr] = y_l[y_idx]
                ptr += 1

        fcolors = [plt.cm.Spectral_r(x) for x in numpy.random.rand(nx * ny)]
        M = numpy.zeros(n_pix) - 20.
        for i in range(n_pix):
            if i in self.y_vals:
                M[i] = numpy.mean(self.y_vals[i])

        min_m = M[numpy.where(M > -10)].min() if min_m == -100 else min_m
        max_m = M.max() if max_m == -100 else max_m

        M = M - min_m
        M = M / (max_m - min_m)
        for i in range(n_pix):
            if M[i] <= 0:
                fcolors[i] = plt.cm.Greys(.5)
            else:
                fcolors[i] = plt.cm.jet(M[i])
        fig_y = ((8. * 0.78 / (self.n_top + 0.5) / 2.) * (3. * ny + 1) / numpy.sqrt(3)) / 0.78
        fig = plt.figure(figsize=(8, fig_y), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        col = collections.RegularPolyCollection(6,
                                                sizes=numpy.zeros(nx * ny) + ((8. * 0.78 / (
                                                        self.n_top + 0.5)) / 2. * 72.) ** 2 * 4. * numpy.pi / 3.,
                                                offsets=list(zip(b_x, b_y)),
                                                transOffset=ax.transData)
        col.set_color(fcolors)
        ax.add_collection(col, autolim=True)
        ax.set_xlim(-0.5, nx)
        ax.set_ylim(-1, nx + 0.5)
        plt.axis('off')
        if cbar:
            figbar = plt.figure(figsize=(8, 1.), dpi=100)
            ax1 = figbar.add_axes([0.05, 0.8, 0.9, 0.15])
            cmap = cm.jet
            norm = mpl.colors.Normalize(vmin=min_m, vmax=max_m)
            cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
            cb.set_label('')
        plt.show()
