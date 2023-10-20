from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
from torch.nn import Linear, Module, Sequential, LSTM, Dropout, ReLU
from scipy.linalg import pinvh
from scipy.interpolate import interp1d

columns = {'theta': 'Mag3 [Platter,Position]',
            's1': 'Sensor1 [Sensor thickness]',
            's2': 'Sensor2 [Sensor thickness]',
            's1_sgn': 'Mag3 [QCM,S1 signal]',
            's2_sgn': 'Mag3 [QCM,S2 signal]',
            's1_bgd': 'Mag3 [QCM,S1 background]',
            's2_bgd': 'Mag3 [QCM,S1 background]',
            's1_factor': 'Mag3 [QCM,S1 factor] (None)',
            's2_factor': 'Mag3 [QCM,S2 factor] (None)'}

def _get_unit(key: str) -> float:
    udict = {'deg': np.pi / 180.0, 'mm': 1e6, 'um': 1e3, 'nm': 1e0}
    for unit_key in udict:
        units = unit_key.split(',')
        for unit in units:
            if unit in key:
                return udict[unit_key]
    return 1.0

def read_hdf(path: str, key: str, name: str) -> np.ndarray:
    """QCM Data extractor and convertor. Read raw QCM sensors read-outs and
    pre-processes the raw time-series.

    Arguments:
        path: Path to a file.
        attr: Attributes name. One of the following keyword arguments:

            * 'theta' : Platter position.
            * 's1', 's2' : Raw 'Sensor1' and 'Sensor2' read-outs.
            * 's1_bgd', 's2_bgd' : Background sensor read-outs.
            * 's1_sgn', 's2_sgn' : Signal read-outs.
            * 's1_factor', 's2_factor' : Factor correction.

    Returns:
        Time-series is SI units.
    """
    df = pd.read_hdf(path, key)
    for attr in df:
        if attr.startswith(name):
            return _get_unit(attr[len(name):]) * df[attr].to_numpy()
        
def read_csv(path: str, name: str) -> np.ndarray:
    """QCM Data extractor and convertor. Read raw QCM sensors read-outs and
    pre-processes the raw time-series.

    Arguments:
        path: Path to a file.
        attr: Attributes name. One of the following keyword arguments:

            * 'theta' : Platter position.
            * 's1', 's2' : Raw 'Sensor1' and 'Sensor2' read-outs.
            * 's1_bgd', 's2_bgd' : Background sensor read-outs.
            * 's1_sgn', 's2_sgn' : Signal read-outs.
            * 's1_factor', 's2_factor' : Factor correction.

    Returns:
        Time-series is SI units.
    """
    df = pd.read_csv(path)
    for attr in df:
        if attr.startswith(name):
            return _get_unit(attr[len(name):]) * df[attr].to_numpy()

def extract_rotations(theta: np.ndarray, data: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    """Integrate signal in a `limits` window for each rotation.

    Args:
        attr : Attribute's name.
        limits : The window bounds (`min`, `max`) in radians.

    Returns:
        Integrated signal.
    """
    qtn, rmd = np.divmod(theta - limits[0], 2.0 * np.pi)
    idxs = (qtn.astype(int) + 1) * np.asarray(rmd < (limits[1] - limits[0]), dtype=int)

    qcm_sum = np.zeros(idxs.max() + 1)
    qcm_cnt = np.zeros(idxs.max() + 1)
    np.add.at(qcm_sum, idxs, data)
    np.add.at(qcm_cnt, idxs, np.ones(idxs.size))
    return np.unique(qtn), (qcm_sum / qcm_cnt)[1:]

def integrate(signal, background, period):
    grad = np.gradient(signal)
    out = np.zeros(grad.size // period + 1)
    np.add.at(out, np.arange(grad.size) // period, grad - background)
    idxs = np.arange(0, grad.size, period)
    return interp1d(idxs, out, 'linear', fill_value='extrapolate')(np.arange(grad.size))

def interpolate(series: np.ndarray, period: int, shift: int) -> np.ndarray:
    grad = np.gradient(series)
    pts = np.arange(shift, grad.size, period)
    return interp1d(pts, grad[pts], 'linear', fill_value='extrapolate')(np.arange(grad.size))

class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, input_size: int, target_size: int, stride: int=1):
        super(WindowDataset, self).__init__()
        X = sliding_window_view(data[0, :-(target_size * stride)], input_size * stride)[:, ::stride]
        Y = sliding_window_view(data[1, (input_size * stride):], target_size * stride)[:, ::stride]
        self.alpha = X[:, [-1]] - X[:, [0]]
        X = X - self.alpha * np.linspace(0, 1, X.shape[1], endpoint=False)
        Y = Y - self.alpha * (1.0 + np.linspace(0, Y.shape[1] / X.shape[1], Y.shape[1], endpoint=False))
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.std = np.std(X, axis=1, keepdims=True)
        self.X = torch.from_numpy((X - self.mean) / self.std).float().view(X.shape + (1,))
        self.Y = torch.from_numpy((Y - self.mean) / self.std).float().view(Y.shape + (1,))

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
    
class AutoregressiveLSTM(Module):
    def __init__(self, input_size, hidden_size, n_layers: int=1, dropout: float=0.5):
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size, self.n_layers = hidden_size, n_layers
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size,
                         num_layers=self.n_layers, batch_first=True,
                         dropout=dropout if self.n_layers > 1 else 0.0)
        self.dense = Sequential(ReLU(), Dropout(dropout), Linear(self.hidden_size, input_size))

    def init_hidden(self, batch_size: int=0):
        weight = next(self.parameters()).data
        if batch_size:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size)
            cell = weight.new(self.n_layers, batch_size, self.hidden_size)
        else:
            hidden = weight.new(self.n_layers, self.hidden_size)
            cell = weight.new(self.n_layers, self.hidden_size)
        return hidden.zero_(), cell.zero_()
    
    def forward(self, input: torch.Tensor, hidden):
        out, hidden = self.lstm(input, hidden)
        return self.dense(out[..., [-1], :]), hidden

    def forecast(self, input: torch.Tensor, hidden, seq_length: int=1):
        x, hidden = self.forward(input, hidden)
        preds = [x,]
        for _ in range(1, seq_length):
            out, hidden = self.lstm(x, hidden)
            x = self.dense(out)
            preds.append(x)
        return torch.cat(preds, dim=-2), hidden

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