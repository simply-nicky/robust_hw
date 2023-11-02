from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import Linear, Module, Sequential, LSTM, Dropout, ReLU
from numpy.lib.stride_tricks import sliding_window_view

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
