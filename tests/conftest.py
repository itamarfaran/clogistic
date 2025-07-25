import numpy as np
import pytest
from scipy.special import expit
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="session")
def fake_data(n: int = 1_000, p: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, p))
    w = rng.uniform(-0.4, 0.4, p)
    y = rng.binomial(1, expit(X @ w))
    return X, y


@pytest.fixture(scope="session")
def breast_cancer_data():
    return load_breast_cancer(return_X_y=True)

