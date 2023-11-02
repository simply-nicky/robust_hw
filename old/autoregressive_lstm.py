from __future__ import annotations
from typing import ClassVar, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn import Linear, Module, Sequential, LSTM, Dropout, ReLU
from numpy.lib.stride_tricks import sliding_window_view

@dataclass
class QCMData():
    """QCM Data extractor and convertor. Read raw QCM sensors read-outs and
    pre-processes the raw time-series.

    Attributes:
        theta : Platten position.
        s1, s2 : Raw 'Sensor1' and 'Sensor2' read-outs.
        s1_bgd, s2_bgd : Background sensor read-outs.
        s1_sgn, s2_sgn : Signal read-outs.
    """
    s1      : np.ndarray
    s2      : np.ndarray
    s1_sgn  : np.ndarray
    s2_sgn  : np.ndarray
    s1_bgd  : np.ndarray
    s2_bgd  : np.ndarray
    theta   : np.ndarray
    dtheta  : Optional[np.ndarray] = None

    columns : ClassVar[Dict[str, str]] = {'theta': 'Mag3 [Platter,Position]',
                                          's1': 'Sensor1 [Sensor thickness]',
                                          's2': 'Sensor2 [Sensor thickness]',
                                          's1_sgn': 'Mag3 [QCM,S1 signal]',
                                          's2_sgn': 'Mag3 [QCM,S2 signal]',
                                          's1_bgd': 'Mag3 [QCM,S1 background]',
                                          's2_bgd': 'Mag3 [QCM,S1 background]'}
    units   : ClassVar[Dict[str, float]] = {'deg': np.pi / 180.0, 'mm': 1e6, 'um': 1e3, 'nm': 1e0}

    def __post_init__(self):
        self.theta = np.unwrap(self.theta)
        if self.dtheta is None:
            self.dtheta = np.gradient(self.theta, edge_order=2)

    @classmethod
    def _get_unit(cls, key: str) -> float:
        for unit_key in cls.units:
            units = unit_key.split(',')
            for unit in units:
                if unit in key:
                    return cls.units[unit_key]
        return 1.0

    @classmethod
    def import_csv(cls, path: str) -> QCMData:
        """Read all the sensor read-outs from a CSV file.

        Args:
            path : Path to a file.

        Returns:
            Updated :class:`QCMData` container.
        """
        df = pd.read_csv(path)
        data = {}
        for col, name in cls.columns.items():
            for attr in df:
                if attr.startswith(name):
                    data[col] = cls._get_unit(attr[len(name):]) * df[attr].to_numpy()
        return cls(**data)

    @classmethod
    def import_hdf(cls, path: str, key: str) -> QCMData:
        """Read all the sensor read-outs from a HDF5 file.

        Args:
            path : Path to a file.
            key : Data key.

        Returns:
            Updated :class:`QCMData` container.
        """
        df = pd.read_hdf(path, key)
        data = {}
        for col, name in cls.columns.items():
            for attr in df:
                if attr.startswith(name):
                    data[col] = cls._get_unit(attr[len(name):]) * df[attr].to_numpy()
        return cls(**data)

    def extract_rotations(self, attr: str, limits: Tuple[float, float]) -> np.ndarray:
        """Integrate signal in a `limits` window for each rotation.

        Args:
            attr : Attribute's name.
            limits : The window bounds (`min`, `max`) in radians.

        Returns:
            Integrated signal.
        """
        qtn, rmd = np.divmod(self.theta - limits[0], 2.0 * np.pi)
        idxs = (qtn.astype(int) + 1) * np.asarray(rmd < (limits[1] - limits[0]), dtype=int)

        qcm_sum = np.zeros(idxs.max() + 1)
        qcm_cnt = np.zeros(idxs.max() + 1)
        np.add.at(qcm_sum, idxs, getattr(self, attr))
        np.add.at(qcm_cnt, idxs, np.ones(idxs.size))
        return (qcm_sum / qcm_cnt)[1:]

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

    def init_hidden(self, batch_size: int=0) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        if batch_size:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size)
            cell = weight.new(self.n_layers, batch_size, self.hidden_size)
        else:
            hidden = weight.new(self.n_layers, self.hidden_size)
            cell = weight.new(self.n_layers, self.hidden_size)
        return hidden.zero_(), cell.zero_()

    def forward(self, input: torch.Tensor, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.lstm(input, hidden)
        return self.dense(out[..., [-1], :]), hidden

    def forecast(self, input: torch.Tensor, hidden, seq_length: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        x, hidden = self.forward(input, hidden)
        preds = [x,]
        for _ in range(1, seq_length):
            out, hidden = self.lstm(x, hidden)
            x = self.dense(out)
            preds.append(x)
        return torch.cat(preds, dim=-2), hidden
