from setuptools import setup, find_packages

setup(
    name="neurostatslib",
    version="0.1",
    description="Flatiron Prototype Package",
    license="None",
    packages=find_packages(),
    install_requires=['jax',
                      'jaxopt',
                      'matplotlib',
                      # for numpy.typing.ArrayLike
                      'numpy>1.20',
                      'scikit-learn',
                      'scipy',
                      'typing_extensions']
)
