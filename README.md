# Robust Holt-Winters smoothing filter
**robust_hw** is a [JAX](https://jax.readthedocs.io/en/latest/#)-powered implementation of Robust Holt-Winters smoothing
filter. Mostly inspired by gradient optimisers in [Optax](https://optax.readthedocs.io/en/latest/).

## Dependencies

- [Python](https://www.python.org/) 3.7 or later (Python 2.x is **not** supported).
- [JAX](https://github.com/google/jax) 0.4.14 or later.
- [h5py](https://www.h5py.org) 3.9.0 or later.
- [NumPy](https://numpy.org) 1.25.2 or later.
- [Optax](https://github.com/google-deepmind/optax) o.1.7 or later.
- [Pandas](https://pandas.pydata.org) 2.0.3 or later.
- [SciPy](https://scipy.org) 1.11.1 or later.
- [tqdm](https://github.com/tqdm/tqdm) 4.66.1 or later.

## Installation
The simplest way to install the requirements is to use *pip*:

    pip install -r requirements.txt

Then, you can install from source with the following command:

    python setup.py install

## Article

The implementation is based on the article "Robust forecasting with exponential and Holt-Winters smoothing" published in [Wiley][https://onlinelibrary.wiley.com/doi/10.1002/for.1125].