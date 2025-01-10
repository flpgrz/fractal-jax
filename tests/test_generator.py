import unittest
import numpy as np
from fractal_jax.generator import FractalJax

class TestFractalJax(unittest.TestCase):

    def setUp(self):
        self.iterations = 100
        self.divergence_threshold = 2.0
        self.backend = 'cpu'
        self.fractal_jax = FractalJax(self.iterations, self.divergence_threshold, self.backend)

    def test_generate_mandelbrot(self):
        x_range = (-2, 1)
        y_range = (-1.5, 1.5)
        pixel_res = 100
        result = self.fractal_jax.generate_mandelbrot(x_range, y_range, pixel_res)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (300, 300))

    def test_generate_julia(self):
        c = complex(-0.7, 0.27015)
        x_range = (-1.5, 1.5)
        y_range = (-1.5, 1.5)
        pixel_res = 100
        result = self.fractal_jax.generate_julia(c, x_range, y_range, pixel_res)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (300, 300))
        assert False

if __name__ == '__main__':
    unittest.main()