import sys

import numpy

from somsphere import Topology
from somsphere.models import DY


def get_index(ix, iy, nx, ny):
    return iy * nx + ix


def get_pair(ii, nx, ny):
    iy = int(numpy.floor(ii / nx))
    ix = ii % nx
    return ix, iy


def get_neighbors(ix, iy, nx, ny, index=False, hex=False):
    """
    Get neighbors for rectangular/hexagonal grid given its
    coordinates and size of grid

    :param int ix: Coordinate in the x-axis
    :param int iy: Coordinate in the y-axis
    :param int nx: Number fo cells along the x-axis
    :param int ny: Number fo cells along the y-axis
    :param bool index: Return indexes in the map format
    :param bool hex: Set the grid to hexagonal
    :return: Array of indexes for direct neighbors
    """
    ns = []

    if ix - 1 >= 0:
        ns.append((ix - 1, iy))
    if iy - 1 >= 0:
        ns.append((ix, iy - 1))
    if ix + 1 < nx:
        ns.append((ix + 1, iy))
    if iy + 1 < ny:
        ns.append((ix, iy + 1))

    even = iy % 2 == 0 or (not hex)
    if even and ix - 1 >= 0 and iy - 1 >= 0:
        ns.append((ix - 1, iy - 1))
    if even and ix - 1 >= 0 and iy + 1 < ny:
        ns.append((ix - 1, iy + 1))
    if not even and ix + 1 < nx and iy - 1 >= 0:
        ns.append((ix + 1, iy - 1))
    if not even and ix + 1 < nx and iy + 1 < ny:
        ns.append((ix + 1, iy + 1))

    ns = numpy.array(ns)
    if not index:
        return ns
    ins = []
    for i in range(len(ns)):
        ins.append(get_index(ns[i, 0], ns[i, 1], nx, ny))
    return numpy.array(ins)


def geometry(topology: Topology, n_top, periodic=False):
    """
    Pre-compute distances between cells in a given topology
    and store it on a dist_lib array

    :param Enum topology: Topology ('grid','hex','sphere')
    :param int n_top: Size of map,  for grid Size=n_top*n_top,
        for hex Size=n_top*(n_top+1[2]) if Ntop is even[odd] and for sphere
        Size=12*n_top*n_top and top must be power of 2
    :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
    :return: 2D array with distances pre computed between cells and total number of units
    :rtype: 2D float array, int
    """
    if topology == Topology.SPHERE:
        try:
            import healpy as hpx
        except:
            print('Error: healpy module not found, use grid or hex topologies')
            sys.exit(0)
        nside = n_top
        n_pix = 12 * nside ** 2
        dist_lib = numpy.zeros((n_pix, n_pix))
        for i in range(n_pix):
            ai = hpx.pix2ang(nside, i)
            for j in range(i + 1, n_pix):
                aj = hpx.pix2ang(nside, j)
                dist_lib[i, j] = hpx.rotator.angdist(ai, aj)
                dist_lib[j, i] = dist_lib[i, j]
        dist_lib[numpy.where(numpy.isnan(dist_lib))] = numpy.pi
    elif topology == Topology.GRID:
        nx = n_top
        ny = n_top
        n_pix = nx * ny
        map_x_y = numpy.mgrid[0:1:complex(0, nx), 0:1:complex(0, ny)]
        map_x_y = numpy.reshape(map_x_y, (2, n_pix))
        b_x = map_x_y[1]
        b_y = map_x_y[0]
        dx = 1. / (nx - 1)
        dy = 1. / (ny - 1)
        dist_lib = numpy.zeros((n_pix, n_pix))
        for i in range(n_pix):
            for j in range(i + 1, n_pix):
                if not periodic:
                    dist_lib[i, j] = numpy.sqrt((b_x[i] - b_x[j]) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    dist_lib[j, i] = dist_lib[i, j]
                else:
                    s0 = numpy.sqrt((b_x[i] - b_x[j]) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    s1 = numpy.sqrt((b_x[i] - (b_x[j] + 1. + dx)) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    s2 = numpy.sqrt((b_x[i] - (b_x[j] + 1. + dx)) ** 2 + (b_y[i] - (b_y[j] + 1. + dy)) ** 2)
                    s3 = numpy.sqrt((b_x[i] - (b_x[j] + 0.)) ** 2 + (b_y[i] - (b_y[j] + 1. + dy)) ** 2)
                    s4 = numpy.sqrt((b_x[i] - (b_x[j] - 1. - dx)) ** 2 + (b_y[i] - (b_y[j] + 1. + dy)) ** 2)
                    s5 = numpy.sqrt((b_x[i] - (b_x[j] - 1. - dx)) ** 2 + (b_y[i] - (b_y[j] + 0.)) ** 2)
                    s6 = numpy.sqrt((b_x[i] - (b_x[j] - 1. - dx)) ** 2 + (b_y[i] - (b_y[j] - 1. - dy)) ** 2)
                    s7 = numpy.sqrt((b_x[i] - (b_x[j] + 0.)) ** 2 + (b_y[i] - (b_y[j] - 1. - dy)) ** 2)
                    s8 = numpy.sqrt((b_x[i] - (b_x[j] + 1. + dx)) ** 2 + (b_y[i] - (b_y[j] - 1. - dy)) ** 2)
                    dist_lib[i, j] = numpy.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
                    dist_lib[j, i] = dist_lib[i, j]
    elif topology == Topology.HEX:
        kk, off = 0, 0.0
        x_l, y_l = numpy.arange(0, n_top, 1.), numpy.arange(0, n_top, DY)
        nx, ny = len(x_l), len(y_l)
        n_pix = nx * ny
        b_x, b_y = numpy.zeros(n_pix), numpy.zeros(n_pix)
        last = ny * DY

        for jj in range(ny):
            for ii in range(nx):
                off = 0. if jj % 2 == 0 else 0.5
                b_x[kk] = x_l[ii] + off
                b_y[kk] = y_l[jj]
                kk += 1

        dist_lib = numpy.zeros((n_pix, n_pix))
        for i in range(n_pix):
            for j in range(i + 1, n_pix):
                if not periodic:
                    dist_lib[i, j] = numpy.sqrt((b_x[i] - b_x[j]) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    dist_lib[j, i] = dist_lib[i, j]
                else:
                    s0 = numpy.sqrt((b_x[i] - b_x[j]) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    s1 = numpy.sqrt((b_x[i] - (b_x[j] + nx)) ** 2 + (b_y[i] - b_y[j]) ** 2)
                    s2 = numpy.sqrt((b_x[i] - (b_x[j] + nx)) ** 2 + (b_y[i] - (b_y[j] + last)) ** 2)
                    s3 = numpy.sqrt((b_x[i] - (b_x[j] + 0)) ** 2 + (b_y[i] - (b_y[j] + last)) ** 2)
                    s4 = numpy.sqrt((b_x[i] - (b_x[j] - nx)) ** 2 + (b_y[i] - (b_y[j] + last)) ** 2)
                    s5 = numpy.sqrt((b_x[i] - (b_x[j] - nx)) ** 2 + (b_y[i] - (b_y[j] + 0)) ** 2)
                    s6 = numpy.sqrt((b_x[i] - (b_x[j] - nx)) ** 2 + (b_y[i] - (b_y[j] - last)) ** 2)
                    s7 = numpy.sqrt((b_x[i] - (b_x[j] + 0)) ** 2 + (b_y[i] - (b_y[j] - last)) ** 2)
                    s8 = numpy.sqrt((b_x[i] - (b_x[j] + nx)) ** 2 + (b_y[i] - (b_y[j] - last)) ** 2)
                    dist_lib[i, j] = numpy.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
                    dist_lib[j, i] = dist_lib[i, j]

    return dist_lib, n_pix


def is_power_2(value):
    """
    Check if passed value is a power of 2
    """
    return value != 0 and ((value & (value - 1)) == 0)


def get_alpha(t, alpha_start, alpha_end, NT):
    """
    Get value of alpha at a given time
    """
    return alpha_start * numpy.power(alpha_end / alpha_start, float(t) / float(NT))


def get_sigma(t, sigma_0, sigma_f, NT):
    """
    Get value of sigma at a given timesure
    """
    return sigma_0 * numpy.power(sigma_f / sigma_0, float(t) / float(NT))


def count_modified_cells(bmu, map_d, sigma):
    """
    Neighborhood function which quantifies how much cells around the best matching one are modified

    :param int bmu: best matching unit
    :param float map_d: array of distances computed with :func:`geometry`
    """
    return numpy.exp(-(map_d[bmu] ** 2) / sigma ** 2)
