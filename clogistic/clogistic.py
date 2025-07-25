"""
Constrained Logistic Regression.


Implementation based on scikit-learn Logistic Regression.

    sklearn.linear_model.LogisticRegression
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import warnings
from typing import Any, Literal, Optional, Union

import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length, compute_class_weight
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

try:
    from sklearn.base import _fit_context
except ImportError:
    # _fit_context introduced in sklearn 1.3.0,
    # clogistic supports sklearn>=1.2.0:
    # this block is to support sklearn>=1.2.0,<1.3.0

    def _fit_context(*, prefer_skip_nested_validation):
        def decorator(fit_method):
            def wrapper(estimator, *args, **kwargs):
                return fit_method(estimator, *args, **kwargs)

            return wrapper

        return decorator


PenaltyType = Optional[Literal["l1", "l2", "elasticnet"]]


def _check_solver(
    solver: Literal["ecos", "scs", "lbfgs"],
    penalty: PenaltyType,
    bounds: Optional[Bounds],
    constraints: Optional[LinearConstraint],
    warm_start: bool,
    l1_ratio: Optional[float],
) -> None:
    if solver == "lbfgs":
        if penalty in ("l1", "elasticnet") and bounds is not None:
            raise ValueError(
                "Solver 'lbfgs' does not support bound constraints with penalty "
                "'l1' and 'elasticnet'; choose either 'ecos' or 'scs' solver."
            )

        if constraints is not None:
            raise ValueError("'lbfgs' solver does not  supports linear constraints.")

        if penalty in ("l1", "elasticnet") and warm_start:
            raise ValueError(
                "Solver 'lbfgs' does not support warm start with 'l1' "
                "and 'elasticnet' regularization."
            )

    if penalty == "elasticnet":
        if not isinstance(l1_ratio, numbers.Number) or not 0 <= l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1; got {}.".format(l1_ratio)
            )
    elif l1_ratio is not None:
        warnings.warn(
            "l1_ratio parameter is only used when penalty "
            "is 'elasticnet'; got penalty={}.".format(penalty)
        )


def _check_bounds(bounds: Bounds, n: int, fit_intercept: bool) -> None:
    if not isinstance(bounds, Bounds):
        raise TypeError("bounds is not of type scipy.optimize.Bounds.")

    lb = bounds.lb
    ub = bounds.ub
    check_consistent_length(lb, ub)

    if n + int(fit_intercept) != len(lb):
        raise ValueError(
            "Length of lower bounds is incorrect; got {} and "
            "must be {}.".format(len(lb), n + int(fit_intercept))
        )


def _check_constraints(
    constraints: LinearConstraint, n: int, fit_intercept: bool
) -> None:
    if not isinstance(constraints, LinearConstraint):
        raise TypeError(
            "constraints is not of type " "scipy.optimize.LinearConstraint."
        )

    A = constraints.A
    lb = constraints.lb
    ub = constraints.ub
    check_consistent_length(lb, ub)

    if n + int(fit_intercept) != A.shape[1]:
        raise ValueError(
            "Number of columns of matrix A is incorrect; got {} "
            "and must be {}.".format(A.shape[1], n + int(fit_intercept))
        )

    check_consistent_length(A, lb)


def _logistic_l1_loss_and_grad(
    w2: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    penalty: PenaltyType,
    fit_intercept: bool,
    l1_ratio: Optional[float],
    sample_weight: Optional[np.ndarray],
) -> tuple[float, np.ndarray]:

    n_samples, n_features = X.shape

    grad = np.empty_like(w2)
    reg = 0.0
    reg_grad = np.zeros(w2.size)

    c = 0.0
    if fit_intercept:
        c = w2[-1]
        w = w2[:n_features] - w2[n_features:-1]
        t = w2[:n_features] + w2[n_features:-1]
    else:
        w = w2[:n_features] - w2[n_features:]
        t = w2[:n_features] + w2[n_features:]

    z = safe_sparse_dot(X, w) + c
    yz = y * z

    if penalty == "l1":
        reg = alpha * t.sum()
        reg_grad = alpha
    elif penalty == "elasticnet":
        regl2 = 0.5 * (1 - l1_ratio) * alpha * np.dot(w, w)
        regl1 = l1_ratio * alpha * t.sum()
        reg = regl2 + regl1
        rg1 = alpha * l1_ratio
        rg2 = alpha * (1 - l1_ratio) * w
        reg_grad[: 2 * n_features] = np.concatenate([rg2, -rg2]) + rg1

    out = np.sum(sample_weight * np.logaddexp(0, -yz)) + reg

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    g = safe_sparse_dot(X.T, z0)

    if fit_intercept:
        grad[:n_features] = g
        grad[n_features:-1] = -g
        grad[-1] = z0.sum()
    else:
        grad[:n_features] = g
        grad[n_features:] = -g

    grad += reg_grad

    return out, grad


def _logistic_loss_and_grad(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    penalty: PenaltyType,
    fit_intercept: bool,
    sample_weight: Optional[np.ndarray],
) -> tuple[float, np.ndarray]:

    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    c = 0.0
    if fit_intercept:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    yz = y * z

    if penalty == "l2":
        reg = 0.5 * alpha * np.dot(w, w)
        reg_grad = alpha * w
    else:
        reg = 0
        reg_grad = 0

    out = np.sum(sample_weight * np.logaddexp(0, -yz)) + reg

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + reg_grad

    if fit_intercept:
        grad[-1] = z0.sum()
    return out, grad


def _fit_lbfgsb(
    penalty: PenaltyType,
    tol: float,
    C: float,
    fit_intercept: bool,
    max_iter: int,
    l1_ratio: Optional[float],
    warm_start_coef: Optional[np.ndarray],
    verbose: bool,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    bounds: Optional[Bounds] = None,
) -> tuple[np.ndarray, float]:

    m, n = X.shape

    mask = y == 1
    y_bin = np.ones(y.shape, dtype=X.dtype)
    y_bin[~mask] = -1.0

    if penalty in ("l1", "elasticnet"):
        func = _logistic_l1_loss_and_grad
        w0 = np.zeros(2 * n + int(fit_intercept))
        bounds = [(0, np.inf)] * n * 2 + [(-np.inf, np.inf)] * int(fit_intercept)
        args = (X, y_bin, 1.0 / C, penalty, fit_intercept, l1_ratio, sample_weight)
    else:
        func = _logistic_loss_and_grad
        w0 = np.zeros(n + int(fit_intercept))
        args = (X, y_bin, 1.0 / C, penalty, fit_intercept, sample_weight)

    if warm_start_coef is not None:
        w0 = warm_start_coef

    options = {"disp": verbose, "gtol": tol, "maxiter": max_iter}

    res = minimize(
        func,
        w0.flatten(),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        args=args,
        options=options,
    )

    if fit_intercept:
        intercept_ = res.x[-1]
        if penalty in ("l1", "elasticnet"):
            coef_ = res.x[:n] - res.x[n:-1]
        else:
            coef_ = res.x[:-1]
    else:
        intercept_ = 0
        if penalty in ("l1", "elasticnet"):
            coef_ = res.x[:n] - res.x[n:]
        else:
            coef_ = res.x

    return coef_, intercept_


def _fit_cvxpy(
    solver: Literal["ecos", "scs"],
    penalty: PenaltyType,
    tol: float,
    C: float,
    fit_intercept: bool,
    max_iter: int,
    l1_ratio: Optional[float],
    warm_start_coef: Optional[np.ndarray],
    verbose: bool,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    bounds: Optional[Bounds] = None,
    constraints: Optional[LinearConstraint] = None,
) -> tuple[np.ndarray, float]:

    m, n = X.shape

    # Decision variables
    if fit_intercept:
        nn = n + 1
        beta = cp.Variable(nn)
        Xbeta = np.c_[X, np.ones(m)] @ beta
    else:
        nn = n
        beta = cp.Variable(nn)
        Xbeta = X @ beta

    # Objective function
    log_likelihood = C * sample_weight @ (cp.multiply(y, Xbeta) - cp.logistic(Xbeta))

    # Bounds
    cons = []
    if bounds is not None:
        lb = bounds.lb
        ub = bounds.ub

        mask_lb = np.isfinite(lb)
        mask_ub = np.isfinite(ub)

        if np.any(mask_lb):
            cons.append(lb[mask_lb] <= beta[mask_lb])

        if np.any(mask_ub):
            cons.append(beta[mask_ub] <= ub[mask_ub])

    # Constraints
    if constraints is not None:
        A = constraints.A
        lb = constraints.lb
        ub = constraints.ub
        Abeta = A @ beta

        mask_lb = np.isfinite(lb)
        mask_ub = np.isfinite(ub)

        if np.any(mask_lb):
            cons.append(lb[mask_lb] <= Abeta[mask_lb])

        if np.any(mask_ub):
            cons.append(Abeta[mask_ub] <= ub[mask_ub])

    # Regularization
    w = beta
    if fit_intercept:
        w = beta[:-1]

    if penalty == "l1":
        penalty = cp.norm(w, 1)
    elif penalty == "l2":
        penalty = 0.5 * cp.sum_squares(w)
    elif penalty == "elasticnet":
        penaltyl2 = 0.5 * (1 - l1_ratio) * cp.sum_squares(w)
        penaltyl1 = l1_ratio * cp.norm(w, 1)
        penalty = penaltyl2 + penaltyl1
    else:
        penalty = 0

    warm_start = False
    if warm_start_coef is not None:
        beta.value = warm_start_coef[0]
        warm_start = True

    obj = cp.Maximize(log_likelihood - penalty)
    problem = cp.Problem(obj, cons)

    if solver == "ecos":
        solve_options = {"solver": cp.ECOS, "abstol": tol}
    elif solver == "scs":
        solve_options = {"solver": cp.SCS, "eps": tol}
    else:
        raise ValueError("Unknown solver {}".format(solver))

    problem.solve(
        max_iters=max_iter,
        verbose=bool(verbose),
        warm_start=warm_start,
        **solve_options,
    )

    if fit_intercept:
        intercept_ = beta.value[-1]
        coef_ = beta.value[:-1]
    else:
        intercept_ = 0
        coef_ = beta.value

    return coef_, intercept_


class ConstrainedLogisticRegression(LogisticRegression):
    """
    Constrained Logistic Regression (aka logit, MaxEnt) classifier.

    This class implements regularized logistic regression supported bound
    and linear constraints using the 'ecos', 'scs' and 'lbfgs' solvers.

    All solvers support only L1, L2 and Elastic-Net regularization or no
    regularization. The 'lbfgs' solver supports bound constraints for L2
    regularization. The 'ecos' and 'scs' solver support bound constraints and
    linear constraints for all regularizations.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Used to specify the norm used in the penalization. The 'lbfgs',
        solver supports only 'l2' penalties if bounds are provided.
        If 'none', no regularization is applied.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    solver : {'ecos', 'lbfgs', 'scs'}, default='ecos'
        Algorithm/solver to use in the optimization problem.

        - Unconstrained 'lbfgs' handles all regularizations.
        - Bound constrained 'lbfgs' handles L2 or no penalty.
        - For other cases, use 'ecos' or 'scs'.

        Note that 'ecos' and 'scs' are general-purpose solvers called via
        CVXPY.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'`. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features)
        Coefficient of the features in the decision function.

    intercept_ : ndarray of shape (1,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    References
    ----------

    lbfgs -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/L-BFGS-Bb.html
    """

    _parameter_constraints: dict = {
        "penalty": [StrOptions({"l1", "l2", "elasticnet"}), None],
        "tol": [Interval(numbers.Real, 0, None, closed="left")],
        "C": [Interval(numbers.Real, 0, None, closed="right")],
        "fit_intercept": ["boolean"],
        "class_weight": [dict, StrOptions({"balanced"}), None],
        "solver": [StrOptions({"ecos", "lbfgs", "scs"})],
        "max_iter": [Interval(numbers.Integral, 0, None, closed="left")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both"), None],
    }

    def __init__(
        self,
        penalty: PenaltyType = "l2",
        *,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        class_weight: Optional[Union[dict[Any, float], Literal["balanced"]]] = None,
        solver: Literal["ecos", "scs", "lbfgs"] = "ecos",
        max_iter: int = 100,
        verbose: Union[int, bool] = 0,
        warm_start: bool = False,
        l1_ratio: Optional[float] = None,
    ):
        super().__init__(
            penalty=penalty,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            l1_ratio=l1_ratio,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        bounds: Optional[Bounds] = None,
        constraints: Optional[LinearConstraint] = None,
    ) -> "ConstrainedLogisticRegression":
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        bounds : scipy.optimize.Bounds or None, default None
            Bounds on the coefficients and intercept.

        constraints : scipy.optimize.LinearConstraint or None, default None
            Linear constraints on the coefficients and intercept.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._validate_params()

        _check_solver(
            solver=self.solver,
            penalty=self.penalty,
            bounds=bounds,
            constraints=constraints,
            warm_start=self.warm_start,
            l1_ratio=self.l1_ratio,
        )

        X, y = self._validate_data(X, y, accept_sparse="csr", order="C")

        if type_of_target(y) != "binary":
            raise ValueError("This solver needs a binary target.")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r" % self.classes_[0]
            )

        if bounds is not None:
            _check_bounds(bounds, n_features, self.fit_intercept)

        if constraints is not None:
            _check_constraints(constraints, n_features, self.fit_intercept)

        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if self.class_weight is not None:
            le = LabelEncoder()
            class_weight_ = compute_class_weight(
                class_weight=self.class_weight, classes=self.classes_, y=y
            )
            sample_weight *= class_weight_[le.fit_transform(y)]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
            )

        if self.solver in ("ecos", "scs"):
            coef_, intercept_ = _fit_cvxpy(
                solver=self.solver,
                penalty=self.penalty,
                tol=self.tol,
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                l1_ratio=self.l1_ratio,
                warm_start_coef=warm_start_coef,
                verbose=self.verbose,
                X=X,
                y=y,
                sample_weight=sample_weight,
                bounds=bounds,
                constraints=constraints,
            )
        else:
            coef_, intercept_ = _fit_lbfgsb(
                penalty=self.penalty,
                tol=self.tol,
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                l1_ratio=self.l1_ratio,
                warm_start_coef=warm_start_coef,
                verbose=self.verbose,
                X=X,
                y=y,
                sample_weight=sample_weight,
                bounds=bounds,
            )

        self.coef_ = np.asarray([coef_])
        self.intercept_ = np.asarray([intercept_])
        self.n_iter_ = np.zeros((len(self.classes_),), dtype=np.int32)

        return self

    def to_logistic_regression(self) -> LogisticRegression:
        check_is_fitted(self)
        new = LogisticRegression()
        for k, v in self.__dict__.items():
            setattr(new, k, v)
        return new
