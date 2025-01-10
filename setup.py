from setuptools import setup, find_packages


def description():
    description = (
        "Generate fractals with JAX."
    )
    return description


install_requires = [
    "matplotlib"
]

extras_require = {
    'test': ['pytest', 'pylint']
}

setup(
    name='fractal_jax',
    version='0.2.1',
    description=description(),
    author="Filippo Grazioli",
    author_email="sendtofilippo@gmail.com",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
)
