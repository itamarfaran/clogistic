import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="session")
def fake_data(n: int = 1_000, p: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, p))
    w = rng.uniform(-0.4, 0.4, p)
    y = rng.binomial(1, expit(X @ w))
    return X, y, w


@pytest.fixture(scope="session")
def breast_cancer_data():
    return load_breast_cancer(return_X_y=True)


@pytest.fixture(scope="session")
def breast_cancer_dataframe(breast_cancer_data):
    X, y = breast_cancer_data
    X = pd.DataFrame(X, columns=["feature_{}".format(i) for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y
