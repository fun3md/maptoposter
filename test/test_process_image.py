import unittest
import os
import tempfile
import numpy as np
import cv2
from process_image import main

class TestProcessImage(unittest.TestCase):
    def setUp(self):
        # Create temporary directory and test images
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.temp_dir, "input.png")
        self.output_path = os.path.join(self.temp_dir, "output.png")
        # Create a simple gradient test image
        gradient = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        cv2.imwrite(self.input_path, gradient)

    def tearDown(self):
        # Clean up temporary files
        for path in [self.input_path, self.output_path]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self.temp_dir)

    def test_import_and_basic_execution(self):
        """Test that process_image can be imported and executed without errors."""
        # Ensure the module can be imported
        import process_image
        self.assertIsNotNone(process_image.main)
        # Execute main with our test image
        import sys
        original_argv = sys.argv
        sys.argv = ['process_image.py', self.input_path, self.output_path]
        try:
            # Capture stdout to suppress output
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull
                process_image.main()
        finally:
            sys.argv = original_argv
            sys.stdout = sys.__stdout__
        # Verify that output file was created
        self.assertTrue(os.path.exists(self.output_path))

if __name__ == '__main__':
    unittest.main()