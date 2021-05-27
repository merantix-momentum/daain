from abc import ABC

import numpy as np
import scipy
from pyod.models.hbos import HBOS

from mxlabs_ood_detection.utils.array_utils import bflatten, bmean_squared_over_voxels


class Classifier(ABC):
    """Base class for the classifiers below"""

    def __init__(self, training_data=None, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class MahalanobisDistance(Classifier):
    def __init__(self, training_data):
        self.inverse_covariance = np.linalg.inv(np.cov(bflatten(training_data).T))
        self.mean = np.mean(training_data)

    def __call__(self, data):
        x = bflatten(data) - self.mean
        return np.dot(np.dot(x, self.inverse_covariance), x.T).diagonal()


class EuclideanDistance(Classifier):
    def __init__(self, training_data=None):
        pass

    def __call__(self, data):
        return np.mean(bflatten(data) ** 2, axis=1)


class HarmonicMeanDistance(Classifier):
    def __init__(self, training_data=None):
        pass

    def __call__(self, data):
        return scipy.stats.hmean(bflatten(data) ** 2, axis=1)


class HistogramBasedDistance(Classifier):
    def __init__(
        self, training_data=None, n_bins=40, contamination=0.001, aggregation_method=bmean_squared_over_voxels
    ):
        self.hbos = HBOS(n_bins=n_bins, contamination=contamination)
        if training_data is not None:
            self.hbos.fit(bflatten(training_data))

        self.aggregation_method = aggregation_method

    def __call__(self, data):
        return self.aggregation_method(self.hbos.predict_proba(bflatten(data))[:, 1])

    def __str__(self):
        return "-".join(
            (
                self.__class__.__name__,
                str(self.hbos.n_bins),
                str(self.hbos.contamination),
                self.aggregation_method.__name__,
            )
        )
