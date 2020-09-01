"""
.. module:: SOMZ
.. moduleauthor:: Matias Carrasco Kind

"""
from __future__ import print_function

from builtins import object
from builtins import range
from builtins import zip

__author__ = 'Matias Carrasco Kind'

import numpy
import copy
import sys, os, random
import warnings
from somsphere.models import Topology, DY, SomType

from somsphere.utils import get_sigma, is_power_2, geometry, count_modified_cells, get_alpha

warnings.simplefilter("ignore", RuntimeWarning)
try:
    import somF

    SF90 = True
except:
    SF90 = False


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
    :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
    :param dict dict_dim: dictionary with attributes names
    :param float alpha_start: Initial value of alpha
    :param float alpha_end: End value of alpha
    :param str importance: Path to the file with importance ranking for attributes, default is none
    """

    def __init__(self, X, Y, topology='grid', som_type='online', n_top=28, n_iter=30, alpha_start=0.8, alpha_end=0.5,
                 periodic=False, importance=None, dict_dim=None):
        if topology == 'sphere' and not is_power_2(n_top):
            print('Error, n_top must be power of 2')
            sys.exit(0)

        self.SF90 = SF90
        self.np, self.n_dim = numpy.shape(X)
        self.X = X
        self.Y = Y
        self.topology: Topology = Topology(topology)
        self.som_type: SomType = SomType(som_type)
        self.n_top = n_top
        self.n_iter = n_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.periodic = periodic
        self.dict_dim = dict_dim
        self.dist_lib, self.n_pix = geometry(self.topology, self.n_top, periodic=self.periodic)

        importance = numpy.ones(self.n_dim) if importance is None else importance
        self.importance = importance / numpy.sum(importance)

        self.weights = (numpy.random.rand(self.n_dim, self.n_pix)) + self.X[0][
            0]

    def get_best_cell(self, inputs, return_vals=1):
        """
        Return the closest cell to the input object
        It can return more than one value if needed
        """
        activations = numpy.sum(numpy.transpose([self.importance]) * (
                numpy.transpose(numpy.tile(inputs, (self.n_pix, 1))) - self.weights) ** 2, axis=0)

        return numpy.argmin(activations) if return_vals == 1 else numpy.argsort(activations)[0:return_vals], activations

    def create_map_f(self, inputs_weights=None, evol=False):
        """
        This functions actually create the maps, it uses
        random values to initialize the weights
        It uses a Fortran subroutine compiled with f2py
        """
        if not self.SF90:
            print('Fortran module somF not found, use create_map instead or try' \
                  ' f2py -c -m somF som.f90')
            sys.exit(0)

        self.weights = inputs_weights if inputs_weights is not None else self.weights
        weights_t = None
        if self.som_type == SomType.ONLINE:
            weights_t = somF.map(self.X, self.n_dim, self.n_iter, self.dist_lib, self.np, self.weights,
                                      self.importance, self.n_pix, self.alpha_start, self.alpha_end)
        elif self.som_type == SomType.BATCH:
            weights_t = somF.map_b(self.X, self.n_dim, self.n_iter, self.dist_lib, self.np, self.weights,
                                        self.importance, self.n_pix)
        self.weights = copy.deepcopy(weights_t)

    def create_map(self, inputs_weights=None, random_order=True, evol=False):
        """
        This is same as above but uses python routines instead
        """
        self.weights = inputs_weights if inputs_weights is not None else self.weights
        NT = self.n_iter * self.np
        tt = 0
        sigma0 = self.dist_lib.max()
        sigma_single = numpy.min(self.dist_lib[numpy.where(self.dist_lib > 0.)])
        for it in range(self.n_iter):
            sigma = get_sigma(tt, sigma0, sigma_single, NT)
            if self.som_type == SomType.ONLINE:
                # get alpha, sigma
                alpha = get_alpha(tt, self.alpha_start, self.alpha_end, NT)
                index_random = random.sample(range(self.np), self.np) if random_order else numpy.arange(self.np)
                for i in range(self.np):
                    tt += 1
                    inputs = self.X[index_random[i]]
                    best, activation = self.get_best_cell(inputs)
                    self.weights += alpha * count_modified_cells(best, self.dist_lib, sigma) * numpy.transpose(
                        (inputs - numpy.transpose(self.weights)))

            elif self.som_type == SomType.BATCH:
                # get alpha, sigma
                accum_w = numpy.zeros((self.n_dim, self.n_pix))
                accum_n = numpy.zeros(self.n_pix)
                for i in range(self.np):
                    tt += 1
                    inputs = self.X[i]
                    best, activation = self.get_best_cell(inputs)
                    for kk in range(self.n_dim):
                        accum_w[kk, :] += count_modified_cells(best, self.dist_lib, sigma) * inputs[kk]
                    accum_n += count_modified_cells(best, self.dist_lib, sigma)
                for kk in range(self.n_dim):
                    self.weights[kk] = accum_w[kk] / accum_n

            if evol:
                self.evaluate_map()
                self.save_map(itn=it)

    def evaluate_map(self, input_x=None, input_y=None):
        """
        This functions evaluates the map created using the input Y or a new Y (array of labeled attributes)
        It uses the X array passed or new data X as well, the map doesn't change

        :param float input_x: Use this if another set of values for X is wanted using
            the weigths already computed
        :param float input_y: One  dimensional array of the values to be assigned to each cell in the map
            based on the in-memory X passed
        """
        self.y_vals = {}
        self.i_vals = {}
        in_x = self.X if input_x is None else input_x
        in_y = self.Y if input_y is None else input_y
        for i in range(len(in_x)):
            inputs = in_x[i]
            best, activation = self.get_best_cell(inputs)
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
        bests, _ = self.get_best_cell(line, return_vals=10)
        if best:
            return bests[0]
        for ib in range(10):
            if bests[ib] in self.y_vals:
                return self.y_vals[bests[ib]]
        return numpy.array([-1.])

    def save_map(self, itn=-1, fileout='SOM', path=None):
        """
        Saves the map

        :param int itn: Number of map to be included on path, use -1 to ignore this number
        :param str fileout: Name of output file
        :param str path: path for the output file
        """
        path = os.getcwd() + '/' if path is None else path
        if not os.path.exists(path):
            os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        numpy.save(path + fileout, self)

    def save_map_dict(self, path='', fileout='SOM', itn=-1):
        """
        Saves the map in dictionary format

        :param int itn: Number of map to be included on path, use -1 to ignore this number
        :param str fileout: Name of output file
        :param str path: path for the output file
        """
        SOM = {'W': self.weights, 'y_vals': self.y_vals, 'i_vals': self.i_vals, 'topology': self.topology,
               'n_top': self.n_top, 'n_pix': self.n_pix}
        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn > 0:
            ff = '_%04d' % itn
            fileout += ff
        numpy.save(path + fileout, SOM)

    def plot_map(self, min_m=-100, max_m=-100, colorbar=True):
        """
        Plots the map after evaluating, the cells are colored with the mean value inside each
        one of them

        :param float min_m: Lower limit for coloring the cells, -100 uses min value
        :param float max_m: Upper limit for coloring the cells, -100 uses max value
        :param str colorbar: Include a colorbar ('yes','no')
        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.cm as cm
        from matplotlib import collections
        from matplotlib.colors import colorConverter

        if self.topology == Topology.SPHERE:
            import healpy as hp
            M = numpy.zeros(self.n_pix) + hp.UNSEEN
            for i in range(self.n_pix):
                if i in self.y_vals:
                    M[i] = numpy.mean(self.y_vals[i])
            plt.figure(10, figsize=(8, 8), dpi=100)
            if min_m == -100:
                min_m = M[numpy.where(M > -10)].min()
            if max_m == -100:
                max_m = M.max()
            hp.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=colorbar)

        if self.topology == Topology.GRID:
            M = numpy.zeros(self.n_pix) - 20.
            for i in range(self.n_pix):
                if i in self.y_vals:
                    M[i] = numpy.mean(self.y_vals[i])
            M2 = numpy.reshape(M, (self.n_top, self.n_top))
            plt.figure(figsize=(8, 8), dpi=100)
            if min_m == -100:
                min_m = M2[numpy.where(M2 > -10)].min()
            if max_m == -100:
                max_m = M2.max()
            SM2 = plt.imshow(M2, origin='center', interpolation='nearest', cmap=cm.jet, vmin=min_m, vmax=max_m)
            SM2.cmap.set_under("grey")
            if colorbar:
                plt.colorbar()
            plt.axis('off')
        if self.topology == Topology.HEX:
            x_l, y_l = numpy.arange(0, self.n_top, 1.), numpy.arange(0, self.n_top, DY)
            nx, ny = len(x_l), len(y_l)
            n_pix = nx * ny
            b_x, b_y = numpy.zeros(n_pix), numpy.zeros(n_pix)
            kk = 0
            for jj in range(ny):
                for ii in range(nx):
                    if jj % 2 == 0: off = 0.
                    if jj % 2 == 1: off = 0.5
                    b_x[kk] = x_l[ii] + off
                    b_y[kk] = y_l[jj]
                    kk += 1
            xyo = list(zip(b_x, b_y))
            sizes_2 = numpy.zeros(nx * ny) + ((8. * 0.78 / (self.n_top + 0.5)) / 2. * 72.) ** 2 * 4. * numpy.pi / 3.
            M = numpy.zeros(n_pix) - 20.
            fcolors = [plt.cm.Spectral_r(x) for x in numpy.random.rand(nx * ny)]
            for i in range(n_pix):
                if i in self.y_vals:
                    M[i] = numpy.mean(self.y_vals[i])
            if max_m == -100:
                max_m = M.max()
            if min_m == -100:
                min_m = M[numpy.where(M > -10)].min()
            M = M - min_m
            M = M / (max_m - min_m)
            for i in range(n_pix):
                if M[i] <= 0:
                    fcolors[i] = plt.cm.Greys(.5)
                else:
                    fcolors[i] = plt.cm.jet(M[i])
            figy = ((8. * 0.78 / (self.n_top + 0.5) / 2.) * (3. * ny + 1) / numpy.sqrt(3)) / 0.78
            fig3 = plt.figure(figsize=(8, figy), dpi=100)
            # fig3.subplots_adjust(left=0,right=1.,top=1.,bottom=0.)
            a = fig3.add_subplot(1, 1, 1)
            col = collections.RegularPolyCollection(6, sizes=sizes_2, offsets=xyo, transOffset=a.transData)
            col.set_color(fcolors)
            a.add_collection(col, autolim=True)
            a.set_xlim(-0.5, nx)
            a.set_ylim(-1, nx + 0.5)
            plt.axis('off')
            if colorbar:
                figbar = plt.figure(figsize=(8, 1.), dpi=100)
                ax1 = figbar.add_axes([0.05, 0.8, 0.9, 0.15])
                cmap = cm.jet
                norm = mpl.colors.Normalize(vmin=min_m, vmax=max_m)
                cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
                cb1.set_label('')

        plt.show()
