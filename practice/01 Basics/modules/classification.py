import numpy as np

from modules.metrics import *
from modules.utils import z_normalize


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier
    """

    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        self.X_train = X_train
        self.Y_train = Y_train
        return self

    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """Compute distance between train and test samples"""

        normalize = self.metric_params.get('normalize', False)

        if normalize:
            x_train = z_normalize(x_train)
            x_test = z_normalize(x_test)

        if self.metric == 'euclidean':
            dist = ED_distance(x_train, x_test)
        elif self.metric == 'dtw':
            r = self.metric_params.get('r', 0.05)
            dist = DTW_distance(x_train, x_test, r)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return dist

    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """Find k nearest neighbors for a test sample"""

        distances = []
        for i, x_train in enumerate(self.X_train):
            d = self._distance(x_train, x_test)
            distances.append((d, self.Y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.n_neighbors]

        return neighbors

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels for the test set"""

        y_pred = []
        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)

            neighbor_labels = [label for _, label in neighbors]

            counts = np.bincount(neighbor_labels)
            pred_label = np.argmax(counts)

            y_pred.append(pred_label)

        return np.array(y_pred)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score

    Parameters
    ----------
    y_true: ground truth (correct) labels
    y_pred: predicted labels returned by a classifier

    Returns
    -------
    score: accuracy classification score
    """

    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score
