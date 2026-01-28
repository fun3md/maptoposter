import unittest
import numpy as np
from lut_utils import load_lut, invert_lut, interpolate_lut

class TestLutUtils(unittest.TestCase):
    def test_load_lut_basic(self):
        # Load a sample LUT file from the luts directory
        lut_func = load_lut('luts/laser_correction.csv')
        self.assertTrue(callable(lut_func))
        # Test that the function can interpolate a value
        result = lut_func(0.5)
        self.assertIsInstance(result, (float, np.floating))

    def test_invert_lut(self):
        lut_func = load_lut('luts/laser_correction.csv')
        inv_func = invert_lut(lut_func)
        self.assertTrue(callable(inv_func))
        # Test that the inverted function returns a numeric value
        val = lut_func(0.2)
        inv_val = inv_func(val)
        self.assertIsInstance(inv_val, (float, np.floating))

    def test_interpolate_lut(self):
        lut_func = load_lut('luts/laser_correction.csv')
        result = interpolate_lut(lut_func, 0.75)
        self.assertIsInstance(result, (float, np.floating))

if __name__ == '__main__':
    unittest.main()