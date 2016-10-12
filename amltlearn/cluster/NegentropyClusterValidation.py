"""Negentropy Clustering Validation"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 12/10/2016 21:44

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin


class NegentropyClusterValidation(BaseEstimator, ClusterMixin,
                        TransformerMixin):
    """Negentropy Clustering Validation"""

