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
                      'numpy',
                      'scikit-learn',
                      'scipy',
                      'typing_extensions']
)
