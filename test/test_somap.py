import unittest

import somsphere
from somsphere import SOMap
import numpy as np


class TestSOMap(unittest.TestCase):
    def test_init_error(self):
        som = SOMap([], [], n_top=0, topology="sphere")

    def test_create_map(self):
        data = "../resources/SDSS_MGS.train"
        # just read magnitudes and colors
        dx = np.loadtxt(data, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), unpack=True).T
        np.shape(dx)
        # read zspec (or any other extra column)
        dy = np.loadtxt(data, usecols=(0,), unpack=True).T
        np.shape(dy)
        # create an instance
        map = somsphere.SOMap(topology='grid', n_top=15, n_iter=100, periodic=False)
        map.create_map(dx, dy)  # This actually creates the map using only dx

