import itertools

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from clogistic import ConstrainedLogisticRegression
from tests.conftest import fake_data


def bounds_and_constraints(
    p: int,
    fit_intercept: bool = True,
) -> tuple[Bounds, LinearConstraint]:
    if fit_intercept:
        p = p + 1

    lb = np.r_[np.full(p - 1, -1), -np.inf]
    ub = np.r_[np.zeros(p - 1), np.inf]
    bounds = Bounds(lb, ub)

    lb = np.array([0.0])
    ub = np.array([0.5])
    A = np.zeros((1, p))
    constraints = LinearConstraint(A, lb, ub)

    return bounds, constraints


def assert_predictions(
    clf: LogisticRegression, X: np.ndarray, y: np.ndarray, value: float = 0.9
) -> None:
    assert np.all(np.unique(y) == clf.classes_)

    pred = clf.predict(X)
    assert np.mean(pred == y) > value

    probabilities = clf.predict_proba(X)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(X.shape[0]))

    pred = clf.classes_[np.argmax(clf.predict_log_proba(X), axis=1)]
    assert np.mean(pred == y) > value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"penalty": "new_penalty"},
        {"penalty": "elasticnet"},
        {"tol": -1e-3},
        {"fit_intercept": 0},
        {"class_weight": []},
        {"class_weight": "unbalanced"},
        {"solver": "new_solver"},
        {"max_iter": -10},
        {"warm_start": 1},
        {"solver": "lbfgs", "penalty": "l1", "warm_start": True},
        {"solver": "lbfgs", "penalty": "elasticnet", "warm_start": True},
    ],
)
def test_parameters(fake_data, kwargs):
    with pytest.raises(ValueError):
        ConstrainedLogisticRegression(**kwargs).fit(*fake_data)


@pytest.mark.parametrize(
    "clf",
    [
        ConstrainedLogisticRegression(solver="lbfgs", penalty="l1"),
        ConstrainedLogisticRegression(solver="lbfgs", penalty="elasticnet"),
    ],
)
def test_solver(fake_data, clf):
    X, y, _ = fake_data
    bounds, constraints = bounds_and_constraints(X.shape[1])

    with pytest.raises(ValueError):
        clf.fit(X, y, bounds=bounds)

    with pytest.raises(ValueError):
        clf.fit(X, y, constraints=constraints)


def test_target(fake_data):
    X, y, _ = fake_data

    with pytest.raises(ValueError):
        ConstrainedLogisticRegression().fit(X, np.random.randn(y.size))

    with pytest.raises(ValueError):
        ConstrainedLogisticRegression().fit(X, np.ones(y.size))


def test_bounds_and_constraints(fake_data):
    X, y, _ = fake_data
    bounds, constraints = bounds_and_constraints(X.shape[1], fit_intercept=False)

    with pytest.raises(TypeError):
        ConstrainedLogisticRegression(penalty="l2").fit(
            X, y, bounds=[(-np.inf, np.inf)] * X.shape[1]
        )

    with pytest.raises(ValueError):
        ConstrainedLogisticRegression(penalty="l2").fit(X, y, bounds=bounds)

    with pytest.raises(TypeError):
        ConstrainedLogisticRegression().fit(
            X, y, constraints=[constraints.A, constraints.lb, constraints.ub]
        )

    with pytest.raises(ValueError):
        ConstrainedLogisticRegression().fit(X, y, constraints=constraints)


@pytest.mark.parametrize(
    "solver, penalty, fit_intercept",
    itertools.product(
        ("lbfgs", "ecos", "scs"),
        (None, "l1", "l2", "elasticnet"),
        (True, False),
    ),
)
def test_predict_breast_cancer(breast_cancer_data, solver, penalty, fit_intercept):
    X, y = breast_cancer_data

    # Test constrained logistic regression with the breast cancer dataset
    # Test that all solvers with all regularizations score (>0.93) for the training data
    clf = ConstrainedLogisticRegression(solver=solver, penalty=penalty, l1_ratio=0.5)
    clf.fit(X, y)
    assert_predictions(clf, X, y)


@pytest.mark.parametrize(
    "solver, penalty",
    itertools.product(
        ("ecos", "scs"),
        (None, "l1", "l2", "elasticnet"),
    ),
)
def test_predict_breast_cancer_bounds_constraints(breast_cancer_data, solver, penalty):
    # Test constrained logistic regression with the breast cancer dataset
    X, y = breast_cancer_data
    bounds, constraints = bounds_and_constraints(X.shape[1])

    # Test that all solvers with all regularizations score (>0.93) for the training data
    clf = ConstrainedLogisticRegression(solver=solver, penalty=penalty, l1_ratio=0.5)

    clf.fit(X, y, bounds=bounds, constraints=constraints)
    assert_predictions(clf, X, y)


@pytest.mark.parametrize("solver", ("ecos", "scs"))
def test_warm_start(breast_cancer_data, solver):
    X, y = breast_cancer_data
    clf = ConstrainedLogisticRegression(solver=solver, penalty="l2", warm_start=True)

    score1 = clf.fit(X, y).score(X, y)
    score2 = clf.fit(X, y).score(X, y)
    assert score1 == pytest.approx(score2, rel=1e-1)


@pytest.mark.parametrize(
    "solver, class_weight",
    [
        ("lbfgs", "balanced"),
        ("ecos", {0: 1, 1: 5}),
    ],
)
def test_class_weight(breast_cancer_data, solver, class_weight):
    X, y = breast_cancer_data
    clf = ConstrainedLogisticRegression(
        solver=solver, penalty="l2", class_weight=class_weight
    )
    pred = clf.fit(X, y).predict(X)
    assert np.mean(pred == y) > 0.93


def test_as_logistic_regression(fake_data):
    X, y, _ = fake_data

    clf = ConstrainedLogisticRegression()
    clf.fit(X, y)
    lr = clf.as_logistic_regression()

    assert isinstance(lr, LogisticRegression)
    np.testing.assert_allclose(clf.coef_, lr.coef_)
    np.testing.assert_allclose(clf.intercept_, lr.intercept_)
    np.testing.assert_allclose(clf.predict_proba(X), lr.predict_proba(X))
    np.testing.assert_allclose(clf.predict(X), lr.predict(X))

    with pytest.raises(ValueError):
        lr.fit(X, y)  # solver is "ecos"


def test_close_to_unconstrained(fake_data):
    rng = np.random.default_rng(42)
    X, y, w = fake_data
    y_pos = rng.binomial(1, expit(X @ np.abs(w)))

    lr = LogisticRegression()
    clf = ConstrainedLogisticRegression(solver="lbfgs")
    unbounded_intercept = Bounds([0] * (X.shape[1]) + [-np.inf])

    lr.fit(X, y)
    clf.fit(X, y, bounds=unbounded_intercept)
    assert not np.array_equal(clf.coef_, lr.coef_)

    lr.fit(X, y_pos)
    clf.fit(X, y_pos, bounds=unbounded_intercept)
    np.testing.assert_allclose(clf.coef_, lr.coef_, atol=1e-2)
