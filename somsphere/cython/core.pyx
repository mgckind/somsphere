#cython: language_level=3
import numpy

from somsphere.utils import get_best_cell, count_modified_cells, timeit

def create_map_batch(X, n_row, n_col, accum_w, accum_n, sigma, importance, n_pix, dist_lib, weights):
    cdef int i, j, k
    cdef int nr = n_row
    cdef int nc = n_col
    cdef int npix = n_pix
    cdef double sig = sigma
    cdef double[:] imp = importance
    cdef double[:] an = accum_n
    for i in range(nr):
        inputs = X[i]
        best, activation = get_best_cell(inputs=inputs, importance=imp,
                                         weights=weights, n_pix=npix)
        for j in range(nc):
            accum_w[j, :] += count_modified_cells(best, dist_lib, sig) * inputs[j]
        an += count_modified_cells(best, dist_lib, sig)

    for k in range(nc):
        weights[k] = accum_w[k] / an

def create_map_online(X, n_row, alpha, sigma, random_indices, importance, n_pix, dist_lib, weights):
    cdef int i
    cdef int nr = n_row
    cdef int npix = n_pix
    cdef double alp = alp
    cdef double sig = sigma
    cdef double[:] imp = importance
    for i in range(nr):
        inputs = X[random_indices[i]]
        best, activation = get_best_cell(inputs=inputs, importance=imp,
                                         weights=weights, n_pix=npix)
        weights += alp * count_modified_cells(best, dist_lib, sig) * numpy.transpose(
            (inputs - numpy.transpose(weights)))
