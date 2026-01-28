import unittest
import numpy as np
import gamma_utils

class TestGammaUtils(unittest.TestCase):
    def test_gamma_apply_basic(self):
        # Test gamma correction with identity gamma = 1.0
        data = np.array([0, 128, 255], dtype=np.uint8)
        result = gamma_utils.gamma_apply(data, 1.0)
        np.testing.assert_array_equal(result, data)

    def test_gamma_undo_basic(self):
        data = np.array([0, 128, 255], dtype=np.uint8)
        gamma = 2.0
        corrected = gamma_utils.gamma_apply(data, gamma)
        restored = gamma_utils.gamma_undo(corrected, gamma)
        np.testing.assert_array_equal(restored, data)

    def test_linearize_default(self):
        # Test default linearization (inverse gamma 2.2)
        data = np.array([0, 128, 255], dtype=np.uint8)
        result = gamma_utils.linearize(data)
        # Values should be different (non-linear)
        self.assertFalse(np.array_equal(data, result))

    def test_linearize_with_lut(self):
        # Create a simple LUT: identity mapping
        lut = np.arange(256, dtype=np.uint8)
        data = np.array([0, 128, 255], dtype=np.uint8)
        result = gamma_utils.linearize(data, lut=lut)
        np.testing.assert_array_equal(result, data)

if __name__ == '__main__':
    unittest.main()