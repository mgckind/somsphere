import unittest

import core
import numpy as np

import somsphere
from somsphere import SOMap, get_best_cell, count_modified_cells


class TestSOMap(unittest.TestCase):
    def setUp(self) -> None:
        data = "../resources/SDSS_MGS.train"
        self.dx = np.loadtxt(data, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), unpack=True).T
        self.dy = np.loadtxt(data, usecols=(0,), unpack=True).T

    def test_create_map_online(self):
        map = somsphere.SOMap(self.dx, self.dy, topology='grid', n_top=15, n_iter=100, periodic=False)
        map.create_map()

        map = somsphere.SOMap(self.dx, self.dy, topology='sphere', n_top=8, n_iter=100, periodic=False)
        map.create_map()

        map = somsphere.SOMap(self.dx, self.dy, topology='hex', n_top=15, n_iter=100, periodic=False)
        map.create_map()

    def test_create_map_batch(self):
        map = somsphere.SOMap(self.dx, self.dy, topology='grid', n_top=15, n_iter=100, periodic=False, som_type="batch")
        map.create_map()

        map = somsphere.SOMap(self.dx, self.dy, topology='sphere', n_top=8, n_iter=100, periodic=False,
                              som_type="batch")
        map.create_map()

        map = somsphere.SOMap(self.dx, self.dy, topology='hex', n_top=15, n_iter=100, periodic=False, som_type="batch")
        map.create_map()

    def test_get_best_cell(self):
        n_pix = 225
        n_col = 9
        weights = np.random.rand(n_col, n_pix)
        random_indices = np.random.randint(5000, size=5000)
        importance = np.random.rand(n_col)
        inputs = self.dx[random_indices[0]]
        old_best, old_activations = get_best_cell(inputs=inputs, importance=importance, weights=weights,
                                                  n_pix=n_pix, return_vals=1)
        new_best, new_activations = core.get_best_cell(inputs, importance, weights, n_col, n_pix, 1)

        self.assertEqual(0, sum(old_activations - new_activations))
        self.assertEqual(old_best, new_best[0])

    def test_count_modified_cells(self):
        bmu = [1]
        n_pix = 225
        sigma = 0.7
        dist_lib = np.random.rand(n_pix, n_pix)

        old_cmc = count_modified_cells(1, dist_lib, sigma)
        new_cmc = core.count_modified_cells(bmu, dist_lib, sigma)

        np.allclose(old_cmc, new_cmc)
