import unittest
import numpy as np
import cv2
from stress import apply_stress_v2

class TestStress(unittest.TestCase):
    def test_original_image_when_strength_zero(self):
        # Create a simple test image
        img = np.zeros((10, 10), dtype=np.uint8)
        # Apply stress with strength 0
        result = apply_stress_v2(img, strength=0.0)
        # Should return the original image unchanged
        np.testing.assert_array_equal(result, img)

    def test_invalid_samples_raises_valueerror(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_stress_v2(img, samples=0)

    def test_invalid_iterations_raises_valueerror(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_stress_v2(img, iterations=0)

    def test_invalid_strength_raises_valueerror(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_stress_v2(img, strength=-0.1)
        with self.assertRaises(ValueError):
            apply_stress_v2(img, strength=1.5)

    def test_stress_enhancement_works(self):
        # Create a simple test image with gradient
        img = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        # Apply stress with some strength
        result = apply_stress_v2(img, strength=0.5)
        # Result should be different from original
        self.assertFalse(np.array_equal(result, img))

    def test_safe_range_prevents_division_by_zero(self):
        # Test that safe_range is never zero
        img = np.ones((10, 10), dtype=np.uint8)
        result = apply_stress_v2(img, strength=0.5)
        # Ensure result is a valid image
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, img.shape)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))

if __name__ == '__main__':
    unittest.main()