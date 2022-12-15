# Fractal Jax
Generate figures of the Julia and Mandelbrot sets with Jax.

## Install
This package requires Jax - see the [official JAX documentation](https://github.com/google/jax#installation).
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd mandelbrot-jax
pip install .
```

## How to use

```python
from fractal_jax import FractalJax

# specify number of iterations, divergence threshold and backend
m = FractalJax(iterations=50, divergence_threshold=2, backend="gpu")
```
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(
    m.generate_mandelbrot(x_range=[-2, 1], y_range=[-1.5, 1.5], pixel_res=300)
);
```
![Figure 1](figs/mandelbrot-1.png)

You can also adjust the region which you care about and the pixel resolution:
```python
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(
    m.generate_mandelbrot(x_range=[-1, -0.9], y_range=[-.3, -.2], pixel_res=30000))
);
```
![Figure 2](figs/mandelbrot-2.png)

This library also allows you to generate figures of Julia sets:
```python
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(m.generate_julia(-0.5792518264067199 + 0.5448363340450433j, [-1.5, 1.5], [-1.5, 1.5], 300));
```
![Figure 2](figs/julia-1.png)

## Credits
This implementation is based on the analysis made by [jpivarski](https://gist.github.com/jpivarski) in [mandelbrot-on-all-accelerators.ipynb](https://gist.github.com/jpivarski/da343abd8024834ee8c5aaba691aafc7)