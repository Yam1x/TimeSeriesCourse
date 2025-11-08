import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """
    n = len(ts1)
    mu_t1, mu_t2 = np.mean(ts1), np.mean(ts2)
    sigma_t1, sigma_t2 = np.std(ts1), np.std(ts2)
    
    scalar_product = np.sum(ts1 * ts2)
    
    norm_ed_dist = np.sqrt(
        2 * n * (1 - (scalar_product - n * mu_t1 * mu_t2) / (n * sigma_t1 * sigma_t2))
    )
    
    return float(norm_ed_dist)


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    n = len(ts1)

    dtw_dist = np.full((n + 1, n + 1), np.inf, dtype=float)
    dtw_dist[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_dist[i, j] = cost + min(dtw_dist[i - 1, j], dtw_dist[i, j - 1], dtw_dist[i - 1, j - 1])

    return float(dtw_dist[n, n])
