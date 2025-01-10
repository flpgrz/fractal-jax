from typing import Tuple

import jax
from jax._src.typing import Array
import numpy as np


class FractalJax:
    """A class for generating images of Mandelbrot and Julia sets with Jax.
    """
    def __init__(
            self,
            iterations: int,
            divergence_threshold: float,
            backend: str
    ):
        """
            Attributes
            ----------
            iterations : int
                Number of iteration for computing `z = z**2 + c`
            divergence_threshold : int
                If z > `divergence_threshold`, we assume divergence to inf
            backend : str
                Whether to use CPU or GPU for jit
        """
        self.iterations = iterations
        self.divergence_thershold = divergence_threshold
        self.backend = backend

        self._jit_mandelbrot = jax.jit(self._run_mandelbrot_kernel, backend=backend)
        self._jit_julia = jax.jit(self._run_julia_kernel, backend=backend)

    def _run_mandelbrot_kernel(self, c: Array, fractal: Array) -> Array:
        """Run z = z**2 + c.
        In the Mandelbrot case, c is the point of interest, i.e. the pixel.
        """
        z = c
        for i in range(self.iterations):
            z = z ** 2 + c
            diverged = jax.numpy.absolute(z) > self.divergence_thershold
            diverging_now = diverged & (fractal == self.iterations)
            fractal = jax.numpy.where(diverging_now, i, fractal)
        return fractal

    def _run_julia_kernel(self, c: complex, z: Array, fractal: Array) -> Array:
        """Run z = z**2 + c.
        In the Julia case, c is a constant.
        z_0 is the point of interest, i.e. the pixel.
        """
        for i in range(self.iterations):
            z = z ** 2 + c
            diverged = jax.numpy.absolute(z) > self.divergence_thershold
            diverging_now = diverged & (fractal == self.iterations)
            fractal = jax.numpy.where(diverging_now, i, fractal)
        return fractal

    def generate_mandelbrot(self, x_range: Tuple[int], y_range: Tuple[int], pixel_res: int) -> np.ndarray:
        """Generate the image of the Mandelbrot set.

        Parameters
        ----------
        x_range : Tuple[int]
            Min and max on the x-axis in the complex plane
        y_range : Tuple[int]
            Min and max on the y-axis in the complex plane
        pixel_res : int
            Pixel resolution for box x- and y-axis

        Returns
        -------
        numpy.ndarray
            Generated image
        """
        height = int((y_range[1] - y_range[0]) * pixel_res)
        width = int((x_range[1] - x_range[0]) * pixel_res)
        y, x = jax.numpy.ogrid[
               y_range[1]:y_range[0]:height * 1j,
               x_range[0]:x_range[1]:width * 1j
               ]
        c = x + y * 1j
        fractal = jax.numpy.full(c.shape, self.iterations, dtype=jax.numpy.int32)
        return np.asarray(self._jit_mandelbrot(c, fractal).block_until_ready())

    def generate_julia(self, c: complex, x_range: Tuple[int], y_range: Tuple[int], pixel_res: int) -> np.ndarray:
        """Generate an image of the Julia set for a given complex constant `c`.

        Parameters
        ----------
        c : complex
            The c constant which defines the image
        x_range : Tuple[int]
            Min and max on the x-axis in the complex plane
        y_range : Tuple[int]
            Min and max on the y-axis in the complex plane
        pixel_res : int
            Scaling factor for the pixel resolution for box x- and y-axis

        Returns
        -------
        numpy.ndarray
            Generated image
        """
        height = int((y_range[1] - y_range[0]) * pixel_res)
        width = int((x_range[1] - x_range[0]) * pixel_res)
        y, x = jax.numpy.ogrid[
               y_range[1]:y_range[0]:height * 1j,
               x_range[0]:x_range[1]:width * 1j
               ]
        z = x + y * 1j
        fractal = jax.numpy.full(z.shape, self.iterations, dtype=jax.numpy.int32)
        return np.asarray(self._jit_julia(c, z, fractal).block_until_ready())

