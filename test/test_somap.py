import unittest

from somsphere import SOMap


class TestSOMap(unittest.TestCase):
    def test_init_error(self):
        som = SOMap([], [], n_top=0, topology="sphere")

