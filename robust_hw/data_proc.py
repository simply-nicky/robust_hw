from typing import Tuple
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

columns = {'theta': 'Mag3 [Platter,Position]',
            's1': 'Sensor1 [Sensor thickness]',
            's2': 'Sensor2 [Sensor thickness]',
            's1_sgn': 'Mag3 [QCM,S1 signal]',
            's2_sgn': 'Mag3 [QCM,S2 signal]',
            's1_bgd': 'Mag3 [QCM,S1 background]',
            's2_bgd': 'Mag3 [QCM,S2 background]',
            's1_factor': 'Mag3 [QCM,S1 factor] (None)',
            's2_factor': 'Mag3 [QCM,S2 factor] (None)'}

def _get_unit(key: str) -> float:
    udict = {'deg': np.pi / 180.0, 'mm': 1e6, 'um': 1e3, 'nm': 1e0}
    for unit_key, factor in udict.items():
        units = unit_key.split(',')
        for unit in units:
            if unit in key:
                return factor
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

def extract_rotations(theta: np.ndarray, data: np.ndarray,
                      limits: Tuple[float, float] = (1.5 * np.pi, 13 / 6 * np.pi)) -> np.ndarray:
    """Integrate signal in a `limits` window for each rotation.

    Args:
        attr : Attribute's name.
        limits : The window bounds (`min`, `max`) in radians.

    Returns:
        Integrated signal.
    """
    qtn, rmd = np.divmod(theta - limits[0], 2.0 * np.pi)
    idxs = np.asarray(qtn - qtn.min() + 1, dtype=int) \
         * np.asarray(rmd < (limits[1] - limits[0]), dtype=int)

    qcm_sum = np.zeros(idxs.max() + 1)
    qcm_cnt = np.zeros(idxs.max() + 1)
    np.add.at(qcm_sum, idxs, np.nan_to_num(data))
    np.add.at(qcm_cnt, idxs, np.ones(idxs.size))
    return np.unique(qtn - qtn.min()), np.where(qcm_cnt > 0, qcm_sum / qcm_cnt, 0.0)[1:]

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
