from setuptools import setup, find_packages


def description():
    description = (
        "Generate fractals with JAX."
    )
    return description


install_requires = [
    "matplotlib"
]


setup(
    name='fractal_jax',
    version='0.1.0',
    description=description(),
    author="Filippo Grazioli",
    author_email="sendtofilippo@gmail.com",
    install_requires=install_requires,
    packages=find_packages(),
)
