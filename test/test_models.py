import unittest

from somsphere import Topology


class TestModels(unittest.TestCase):
    def test_get_enum(self):
        self.assertEqual(Topology.GRID, Topology("grid"))
        self.assertEqual(Topology.SPHERE, Topology("sphere"))
        self.assertEqual(Topology.HEX, Topology("hex"))

