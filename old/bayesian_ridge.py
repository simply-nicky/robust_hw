from __future__ import annotations
from dataclasses import dataclass
from scipy.linalg import pinvh
from typing import Optional, Tuple

import numpy as np

def fast_logdet(A: np.ndarray):
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld

@dataclass
class BayesianRidge():
    n_iter          : int = 300
    tol             : float = 1e-3
    alpha_1         : float = 1e-6
    alpha_2         : float = 1e-6
    lambda_1        : Optional[np.ndarray] = None
    lambda_2        : Optional[np.ndarray] = None
    thresh_gamma    : float = 1e-9
    alpha_init      : Optional[float] = None
    lambda_init     : Optional[float] = None
    compute_score   : bool = False
    fit_intercept   : bool = True
    verbose         : bool = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray]=None) -> BayesianRidge:
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = np.asarray(X, np.floating), np.asarray(y, np.floating)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=X.dtype)

        if self.fit_intercept:
            X_offset_ = np.average(X, weights=sample_weight, axis=0)
            X = X - X_offset_

            y_offset_ = np.average(y, axis=0, weights=sample_weight)
            y = y - y_offset_

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            sw_matrix = np.sqrt(sample_weight) * np.eye(X.shape[0], X.shape[0])
            X, y = np.dot(sw_matrix, X), np.dot(sw_matrix, y)

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)` is zero
        self.gamma_ = np.ones(X.shape[1])
        self.alpha_ = self.alpha_init
        self.lambda_ = self.lambda_init
        if self.alpha_ is None:
            self.alpha_ = 1.0 / (np.var(y) + eps)
        if self.lambda_ is None:
            self.lambda_ = np.ones(X.shape[1])

        update_sigma = (self._update_sigma if X.shape[0] >= X.shape[1] else self._update_sigma_woodbury)

        lambda_1, lambda_2 = np.full(X.shape[1], 1e-6), np.full(X.shape[1], 1e-6)
        if self.lambda_1 is not None:
            lambda_1[:self.lambda_1.size] = self.lambda_1[:X.shape[1]]
        if self.lambda_2 is not None:
            lambda_2[:self.lambda_2.size] = self.lambda_2[:X.shape[1]]

        keep_lambda = np.ones(X.shape[1], dtype=bool)
        self.sigma_ = update_sigma(X, keep_lambda)
        self.coef_ = self._update_coef(X, y, self.sigma_, keep_lambda)
        rmse_ = np.sum((y - np.dot(X, self.coef_))**2)

        self.score_ = []
        if self.compute_score:
            score = self._lml(X, rmse_, lambda_1, lambda_2)
            self.scores_.append(score)

        for iter_ in range(self.n_iter):
            # Update alpha and lambda
            self.gamma_[keep_lambda] = 1.0 - self.lambda_[keep_lambda] * np.diag(self.sigma_)
            self.lambda_[keep_lambda] = ((self.gamma_[keep_lambda] + 2.0 * lambda_1[keep_lambda]) /
                                         (self.coef_[keep_lambda]**2 + 2.0 * lambda_2[keep_lambda]))
            self.alpha_ = ((X.shape[0] - np.sum(self.gamma_[keep_lambda]) + 2.0 * self.alpha_1) /
                           (rmse_ + 2.0 * self.alpha_2))
            # keep_lambda = self.lambda_ < self.thresh_lambda
            keep_lambda = self.gamma_ > self.thresh_gamma

            if not keep_lambda.any():
                break

            # Update posterior covariance and mean
            sigma_ = update_sigma(X, keep_lambda)
            coef_ = self._update_coef(X, y, sigma_, keep_lambda)
            coef_[~keep_lambda] = 0

            # Check for convergence
            if np.sum(np.abs(coef_ - self.coef_)) < self.tol:
                if self.verbose:
                    print("Converged after %s iterations" % iter_)
                break

            self.coef_, self.sigma_ = coef_, sigma_
            rmse_ = np.sum((y - np.dot(X, self.coef_))**2)

            if self.compute_score:
                score = self._lml(X, rmse_, lambda_1, lambda_2)
                self.scores_.append(score)

        if self.fit_intercept:
            self.intercept_ = y_offset_ - np.dot(X_offset_, self.coef_.T)
        else:
            self.intercept_ = 0.0

        return self


    def predict(self, X: np.ndarray, return_std: bool=False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = np.dot(X, self.coef_.T) + self.intercept_

        if not return_std:
            return y_mean

        # X = X[:, self.lambda_ < self.thresh_lambda]
        X = X[:, self.gamma_ > self.thresh_gamma]
        y_std = np.sqrt(np.sum(np.dot(X, self.sigma_) * X, axis=1) + 1.0 / self.alpha_)
        return y_mean, y_std

    def _update_coef(self, X: np.ndarray, y: np.ndarray, sigma: np.ndarray,
                     keep_lambda: np.ndarray) -> np.ndarray:
        coef_ = np.zeros(X.shape[1], dtype=X.dtype)
        coef_[keep_lambda] = self.alpha_ * np.linalg.multi_dot([sigma, X[:, keep_lambda].T, y])
        return coef_

    def _update_sigma_woodbury(self, X: np.ndarray, keep_lambda: np.ndarray) -> np.ndarray:
        X_keep = X[:, keep_lambda]
        inv_lambda = 1.0 / self.lambda_[keep_lambda].reshape(1, -1)
        sigma_ = pinvh(np.eye(X.shape[0], dtype=X.dtype) / self.alpha_ 
                       + np.dot(X_keep * inv_lambda, X_keep.T))
        sigma_ = np.dot(sigma_, X_keep * inv_lambda)
        sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
        sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / self.lambda_[keep_lambda]
        return sigma_

    def _update_sigma(self, X: np.ndarray, keep_lambda: np.ndarray) -> np.ndarray:
        # See slides as referenced in the docstring note
        # this function is used when n_samples >= n_features and will
        # invert a matrix of shape (n_features, n_features)
        gram = np.dot(X[:, keep_lambda].T, X[:, keep_lambda])
        eye = np.eye(gram.shape[0], dtype=X.dtype)
        sigma_ = pinvh(self.lambda_[keep_lambda] * eye + self.alpha_ * gram)
        return sigma_

    def _lml(self, X: np.ndarray, rmse: np.ndarray, lambda_1: np.ndarray, lambda_2: np.ndarray) -> float:
        score = (lambda_1 * np.log(self.lambda_) - lambda_2 * self.lambda_).sum()
        score += self.alpha_1 * np.log(self.alpha_) - self.alpha_2 * self.alpha_
        score += 0.5 * (fast_logdet(self.sigma_) + X.shape[0] * np.log(self.alpha_)
                        + np.sum(np.log(self.lambda_)))
        score -= 0.5 * (self.alpha_ * rmse + np.sum(self.lambda_ * self.coef_**2))
        return score