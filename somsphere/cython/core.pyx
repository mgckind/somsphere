#cython: language_level=3
import random

import numpy as np
cimport numpy as cnp

from somsphere.utils import timeit

DTYPE = np.float
ctypedef cnp.float_t DTYPE_t
# def create_map_batch(np.ndarray X, int n_iter, int n_row, int n_col, int n_pix, np.ndarray importance,
#                      np.ndarray dist_lib, np.ndarray weights):
#     cdef int t = 0
#     cdef int total_t = n_iter * n_row
#     cdef float sigma_0 = dist_lib.max()
#     cdef float sigma_f = np.min(dist_lib[np.where(dist_lib > 0.)])
#
#     cdef int i
#     cdef float sigma, activation
#     cdef np.ndarray accum_w, accum_n, best, inputs
#
#     for i in range(n_iter):
#         sigma = get_sigma(sigma_f, sigma_0, t, total_t)
#         accum_w = np.zeros((n_col, n_pix))
#         accum_n = np.zeros(n_pix)
#         for j in range(n_row):
#             inputs = X[j]
#             best, activation = get_best_cell(inputs=inputs, importance=importance,
#                                              weights=weights, n_pix=n_pix)
#             for k in range(n_col):
#                 accum_w[k, :] += count_modified_cells(best, dist_lib, sigma) * inputs[k]
#             accum_n += count_modified_cells(best, dist_lib, sigma)
#
#         for l in range(n_col):
#             weights[l] = accum_w[l] / accum_n
#
#         t += n_row
def get_alpha(float alpha_end, float alpha_start, int curr_t, int total_t):
    """
    Get value of alpha at a given time
    """
    return alpha_start * np.power(alpha_end / alpha_start, float(curr_t) / float(total_t))

def get_sigma(float sigma_f, float sigma_0, int curr_t, int total_t):
    """
    Get value of sigma at a given time
    """
    return sigma_0 * np.power(sigma_f / sigma_0, float(curr_t) / float(total_t))

@timeit
def get_best_cell(cnp.ndarray inputs, cnp.ndarray importance, int n_pix, cnp.ndarray weights, int return_vals):
    """
    Return the closest cell to the input object
    It can return more than one value if needed
    """
    cdef cnp.ndarray activation = np.zeros([n_pix, 1], dtype=DTYPE)

    activations = np.sum(np.transpose([importance]) * (np.transpose(np.tile(inputs, (n_pix, 1))) - weights) ** 2, axis=0)

    return np.argmin(activations) if return_vals == 1 else np.argsort(activations)[0:return_vals], activations

def count_modified_cells(int bmu, cnp.ndarray map_d, float sigma):
    """
    Neighborhood function which quantifies how much cells around the best matching one are modified

    :param int bmu: best matching unit
    :param nparray map_d: array of distances computed with :func:`geometry`
    """
    return np.exp(-(map_d[bmu] ** 2) / sigma ** 2)

def create_map_online(cnp.ndarray X, int n_iter, int n_row, int n_col, int n_pix, float alpha_start, float alpha_end,
                      cnp.ndarray importance, cnp.ndarray dist_lib, cnp.ndarray weights, bint random_order):
    cdef int t = 0
    cdef int total_t = n_iter * n_row
    cdef float sigma_0 = dist_lib.max()
    cdef float sigma_f = np.min(dist_lib[np.where(dist_lib > 0.)])

    cdef int i, best
    cdef float alpha, sigma
    cdef list random_indices
    cdef cnp.ndarray inputs = np.zeros([n_col, 1], dtype=DTYPE)
    cdef cnp.ndarray activation = np.zeros([n_pix, 1], dtype=DTYPE)

    for it in range(n_iter):
        alpha = get_alpha(alpha_end, alpha_start, t, total_t)
        sigma = get_sigma(sigma_f, sigma_0, t, total_t)
        random_indices = random.sample(range(n_row), n_row) if random_order else np.arange(
            n_row)

        for i in range(n_row):
            inputs = X[random_indices[i]]
            best, activation = get_best_cell(inputs=inputs, importance=importance,
                                             weights=weights, n_pix=n_pix, return_vals=1)
            weights += alpha * count_modified_cells(best, dist_lib, sigma) * np.transpose(
                (inputs - np.transpose(weights)))
