"""K-Harmonic Means clustering"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import as_float_array
from sklearn.utils.sparsefuncs import mean_variance_axis
import time


class KHarmonicMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Harmonic means Algorithm

    The k-harmonic means algorithm (KHM) follows the same strategy
    that k-means [1]_, differing in the objective function [2]_.

    Paramereters
    ------------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-harmonic means
        algorithm for a single run.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with
        different centroid seeds. The final results will be the best
        output of n_init consecutive runs in terms of inertia.
    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters,
        n_features) and gives the initial centers.
    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is
        "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto"
        chooses "elkan" for dense data and "full" for sparse data.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters
        > 12 million. This corresponds to about 100MB overhead per job
        using double precision.
        True : always precompute distances
        False : never precompute distances
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare
        convergence
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators
    n_jobs : int
        The number of jobs to use for the computation. This works by
        computing each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing
        code is used at all, which is useful for debugging. For n_jobs
        below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    verbose : int, default 0
        Verbosity mode.
    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate
        to center the data first.  If copy_x is True, then the
        original data is not modified. If False, the original data is
        modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and
        then adding the data mean.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    See Also
    --------
    fuzzy k-means : different adaptation of k-means algorithm

    Notes
    -----
    Objective function implemented [2]_:
    .. math::
    Notes
    ------
    The k-means problem is solved using Lloyd's algorithm.
    The average complexity is given by O(k n T), were n is the number
    of samples and T is the number of iteration.
    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)
    In practice, the k-means algorithm is very fast (one of the
    fastest clustering algorithms available), but it falls in local
    minima. That's why it can be useful to restart it several times.
    Distance function d(x,m), implemented for KHMp (see [2]_):
    .. math:: max(||x_i - c_j||, e)
    where the parameter e is to avoid zero denominators.
    Performance function implemented for KHMp was presented in [2]_.
    The paramater p default value was set to 3.5, as that was the best
    value found by Zhang [2]_, who also stated that for high
    dimensionality data ( > 2 dimensions), larger p values could be
    desired.

    References
    ----------
    .. [1] J. MacQuenn. Some methods for classification and analysis
    of multivariate observations. In L. M. LeCam and J. Neyman,
    editors, Proceedings of the Fifth Berkeley Symposium on
    Mathematical Statistics and Probability, volume 1, pages 281-297,
    Berkeley, CA, 1967. University of California Press.
    .. [2] B. Zhang. Generalized k-harmonic means – boosting in
    unsupervised learning. Technical Report HPL-2000-137,
    Hewlett-Packard Labs, 2000.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])

    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, p=3.5, e=1e-8,
                 precompute_distances='auto', verbose=False,
                 random_state=None, n_jobs=1,
                 algorithm='auto'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.p = p
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.e = e

        self.centroid = None
        self.membership = None
        self.n_iter = None
        self.performance = None
        self.labels_ = None
        self.Y_squared_norm_ = None

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""

        X = check_array(X, accept_sparse='csr', dtype=[np.float64,
                                                       np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        """Check consistency of data in array"""

        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.centroid.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        return X

    def fit(self, X, y=None):
        """Compute k-harmonic means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

        """

        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        tol = _tolerance(X, self.tol)

        self.centroid, self.membership, self.performance, self.n_iter = \
            k_harmonic_means(X, n_clusters=self.n_clusters,
                             n_init=self.n_init,
                             max_iter=self.max_iter,
                             verbose=self.verbose, tol=float(tol),
                             random_state=random_state,
                             n_jobs=self.n_jobs)
        self.labels_ = _get_labels(self.membership)
        self.Y_squared_norm_ = row_norms(self.centroid, squared=True)
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each
        sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        y : array-like or sparse matrix

        """

        return self.fit(X).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is
        called the code book and each value returned by `predict` is
        the index of the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples,
        n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.

        """

        check_is_fitted(self, 'centroid')

        X = self._check_test_data(X)
        Y = self.centroid

        x_squared_norm = row_norms(X, squared=True)
        y_squared_norm = row_norms(Y, squared=True)

        L2_distance = np.maximum(euclidean_distances(
            X=X, Y=Y, Y_norm_squared=y_squared_norm, squared=False,
            X_norm_squared=x_squared_norm), self.e)

        L2_p2_distance = L2_distance ** (self.p+2)
        membership = _calc_membership(X, self.centroid, self.p,
                                      L2_p2_distance=L2_p2_distance,
                                      e=self.e)
        return _get_labels(membership)


def _get_labels(membership):
    """Show cluster with higher membership per each sample"""

    return np.argmax(membership, axis=1)


def _tolerance(X, tol):
    """Return a tolerance which is independent of the data set"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def k_harmonic_means(X, n_clusters, n_init=10, max_iter=300,
                     verbose=False, tol=1e-4, random_state=None,
                     n_jobs=1, precompute_distances=False,
                     algorithm='Zhang', p=3.5, e=1e-8):
    """K-harmonic means clustering algorithm.


    Parameters
    ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

    """

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X, copy=True)

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

    # If the distances are precomputed every job will create a
    # matrix of shape
    # (n_clusters, n_samples). To stop KMeans from eating up memory
    #  we only
    # activate this if the created matrix is guaranteed to be under
    #  100MB. 12
    # million entries consume a little under 100MB if they are of
    # type double.
    # if precompute_distances == 'auto':
    #     n_samples = X.shape[0]
    #     precompute_distances = (n_clusters * n_samples) < 12e6
    # elif isinstance(precompute_distances, bool):
    #     pass
    # else:
    #     raise ValueError(
    #         "precompute_distances should be 'auto' or True/False"
    #         ", but a value of %r was passed" %
    #         precompute_distances)

    # if precompute_distances:
    # precompute squared and non-squared norms of data points
    x_squared_norm = row_norms(X, squared=True)

    best_membership = None
    best_performance = None
    best_centroid = None
    best_n_iter = None

    # By the moment just one version the k-harmonic means is
    # implemented
    # if algorithm == 'Zhang':
    kmeans_single = _k_harmonic_means

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store
        # one set of the best results (as opposed to one set per run
        # per thread).
        for it in range(n_init):
            # run a k-means once
            centroid, membership, n_iter, performance = \
                kmeans_single(X, n_clusters, max_iter=max_iter,
                              verbose=verbose,
                              x_squared_norm=x_squared_norm,
                              tol=tol, random_state=random_state,
                              p=p, e=e)
            # determine if these results are the best so far
            if best_performance is None or performance < \
                    best_performance:
                best_membership = membership.copy()
                best_centroid = centroid.copy()
                best_performance = performance
                best_n_iter = n_iter
    else:
        # parallelisation of k-harmonic means runs
        seeds = random_state.randint(np.iinfo(np.int32).max,
                                     size=n_init)
        # Change seed to ensure
        results = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(kmeans_single)(X, n_clusters, max_iter=max_iter,
                                   verbose=verbose,
                                   x_squared_norm=x_squared_norm,
                                   tol=tol, random_state=seed, p=p,
                                   e=e)
            for seed in seeds)
        # Get results with the lowest inertia
        centroid, membership, n_iter, performance = zip(*results)
        best = np.argmin(performance)
        best_membership = membership[best]
        best_performance = performance[best]
        best_centroid = centroid[best]
        best_n_iter = n_iter[best]

    if not sp.issparse(X):
        best_centroid += X_mean

    return best_centroid, best_membership, best_performance, \
        best_n_iter


def _k_harmonic_means(X, n_clusters, max_iter=300, verbose=False,
                      x_squared_norm=None, tol=1e-4,
                      random_state=None, p=3.5, e=1e-8):
    """K-harmonic means clustering algorithm.


    Parameters
    ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

    """

    if x_squared_norm is None:
        x_squared_norm = row_norms(X, squared=True)

    random_state = check_random_state(random_state)

    performance_old = None
    centroid_old = None
    membership_old = None

    # init
    centroid = _init_centroids(X, n_clusters,
                               random_state=random_state)

    # Allocate memory to store the partial results in order to speed
    # up the overall computation cost and pre-calculations
    membership = np.zeros(shape=(X.shape[0], n_clusters),
                          dtype=float)
    # pre-calculations
    y_squared_norm = row_norms(centroid, squared=True)
    # shape (n_instances, n_centroids)
    L2_distance = euclidean_distances(X=X, Y=centroid,
                                      Y_norm_squared=y_squared_norm,
                                      squared=False,
                                      X_norm_squared=x_squared_norm)
    L2_distance = np.maximum(L2_distance, e)
    inv_L2_p_dist = 1 / (L2_distance ** p)
    inv_L2_p2_dist = 1 / (L2_distance ** (p + 2))

    performance = _evaluate_performance(X, centroid, inv_L2_p_dist,
                                        p, e)

    if verbose:
        print('Initialization complete, with initial performance: '
              ''.join(str(performance)))

    # iterations
    iter_ = 0
    performance_increment = np.inf
    while iter_ < max_iter and performance_increment > tol:
        # Save state
        performance_old = performance
        centroid_old = centroid
        membership_old = membership

        # Pre-calculations
        sum_dist_p2 = np.array([np.sum(inv_L2_p2_dist, axis=1)]).T
        sum_dist_p = np.array([(np.sum(inv_L2_p_dist, axis=1))
                               ** 2]).T
        membership = np.divide(inv_L2_p2_dist, sum_dist_p2)
        div_weight = np.divide(sum_dist_p2, sum_dist_p)
        # shape=(n_samples, n_clusters)
        m_k_i_numerator = np.multiply(membership, div_weight)
        # shape=(n_clusters, 1)
        m_k_denominator = np.array([np.sum(m_k_i_numerator,
                                           axis=0)]).T
        # shape=(n_clusters, n_samples, n_features)
        m_k_i_numerator = np.array([np.multiply(X, np.array([
            m_k_i_numerator[:, c]]).T) for c in range(
            m_k_i_numerator.shape[1])])
        # shape=(n_clusters, n_features)
        mk_numerator = np.sum(m_k_i_numerator, axis=1)
        centroid = np.divide(mk_numerator, m_k_denominator)

        # Calculate distances
        y_squared_norm = row_norms(centroid, squared=True)
        L2_distance = np.maximum(euclidean_distances(X=X, Y=centroid,
                                 Y_norm_squared=y_squared_norm,
                                 squared=False,
                                 X_norm_squared=x_squared_norm),
                                 e)
        inv_L2_p_dist = 1 / (L2_distance ** (p))
        inv_L2_p2_dist = 1 / (L2_distance ** (p + 2))

        performance = _evaluate_performance(X, centroid,
                                            inv_L2_p_dist, p, e)
        performance_increment = performance_old - performance

        print('performance old: ')
        print(performance_old)
        print('performance increment: ')
        print(performance_increment)
        print('iter: ')
        print(iter_)

        iter_ += 1

    if performance_increment < 0:
        centroid = centroid_old
        membership = membership_old
        iter_n = iter_-1
        performance = performance_old
    else:
        iter_n = iter_

    return centroid, membership, iter_n, performance


def _evaluate_performance(X, centroid, inv_distance=None, p=3.5,
                          e=1e-8):
    """Evaluate performance of k-harmonic means clustering result.

    Parameters
    ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

    """

    if inv_distance is None:
        # shape (n_instances, n_centroids)
        L2_distance = np.maximum(euclidean_distances(X=X, Y=centroid,
                                                     squared=False),
                                 e)
        inv_distance = 1 / (L2_distance ** p)
    #     print('Was NONE')
    #
    # print('inv_distance')
    # print(inv_distance.shape)
    # print(np.amax(inv_distance))
    # print(np.amin(inv_distance))
    performance = np.sum(centroid.shape[0] / (np.sum(inv_distance,
                                                     axis=1)))
    # print('centroid.shape[0]')
    # print(centroid.shape[0])
    # # print('inv_distance')
    # # print(inv_distance)
    # print((np.sum(inv_distance, axis=1)).shape)
    # print(np.sum(inv_distance, axis=1))
    print('Performance: ')
    print(performance)

    return performance


def _init_centroids(X, k, random_state=None):
    """Compute the initial centroids

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
    k: int
        number of centroids
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    x_squared_norms:  array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    Returns
    -------
    centers: array, shape(k, n_features)

    """

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if n_samples < k:
        raise ValueError("n_samples=%d should be larger than k=%d" %
                         (n_samples, k))

    seeds = random_state.permutation(n_samples)[:k]
    centers = X[seeds]

    # if sp.issparse(centers):
    #     centers = centers.toarray()

    _validate_center_shape(X, k, centers)

    return centers


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


def _calc_membership(X, centroid, p=3.5, L2_p2_distance=None, e=1e-8):
    """Calculate the memberships of instances to existing clusters

    Parameters
    ----------
    X : array-like or matrix, shape (n_samples, n_features)
        The observations to cluster.
    centroid : float ndarray with shape (k, n_features)
        Centroids found when computing K-harmonic means.
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp.
    L2_p2_distance : array-like or matrix, shape (n_instances,
    n_centroids), float, default : None
        Precomputed distances for speeding up the membership
        calculations.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.

    Return
    ------
    membership : float matrix with shape (n_samples, k)

    """

    k = centroid.shape[0]
    n_samples = X.shape[0]
    membership = np.zeros(shape=(n_samples,k), dtype=X.dtype)
    if L2_p2_distance is None:
        L2_distance = np.maximum(euclidean_distances(
            X=centroid, Y=X, squared=False), e)
        L2_p2_distance = L2_distance ** (p + 2)

    distance = 1/L2_p2_distance
    for i in range(n_samples):
        denominator = np.sum(distance[i, :])
        for j in range(k):
            membership[i, j] = distance[i, j] / denominator

    return membership


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    import sklearn.metrics as sm
    import pandas as pd
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    # import some data to play with
    iris = datasets.load_iris()

    # Store the inputs as a Pandas Dataframe and set the column names
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length',
                 'Petal_Width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    # Set the size of the plot
    plt.figure(figsize=(14, 7))

    # Create a colormap
    colormap = np.array(['red', 'lime', 'black'])

    # Plot Sepal
    plt.subplot(1, 2, 1)
    plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets],
                s=40)
    plt.title('Sepal')

    plt.subplot(1, 2, 2)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets],
                s=40)
    plt.title('Petal')

    # K Means Cluster
    model = KHarmonicMeans(n_clusters=3)
    model.fit(x)

    # This is what KMeans thought
    print(model.labels_)

    # View the results
    # Set the size of the plot
    plt.figure(figsize=(14, 7))

    # Create a colormap
    colormap = np.array(['red', 'lime', 'black'])

    # Plot the Original Classifications
    plt.subplot(1, 2, 1)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets],
                s=40)
    plt.title('Real Classification')

    # Plot the Models Classifications
    plt.subplot(1, 2, 2)
    plt.scatter(x.Petal_Length, x.Petal_Width,
                c=colormap[model.labels_], s=40)
    plt.title('K Mean Classification')

    # The fix, we convert all the 1s to 0s and 0s to 1s.
    predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
    print(model.labels_)
    print(predY)

    # View the results
    # Set the size of the plot
    plt.figure(figsize=(14, 7))

    # Create a colormap
    colormap = np.array(['red', 'lime', 'black'])

    # Plot Orginal
    plt.subplot(1, 2, 1)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets],
                s=40)
    plt.title('Real Classification')

    # Plot Predicted with corrected values
    plt.subplot(1, 2, 2)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[predY],
                s=40)
    plt.title('K Mean Classification')
    plt.plot()

    # Performance Metrics
    print('Performance Metrics')
    print(sm.accuracy_score(y, predY))

    # Confusion Matrix
    print('Confusion Matrix')
    print(sm.confusion_matrix(y, predY))

    # X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    # k_means = KHarmonicMeans(n_clusters=3).fit(X)
    # print(k_means.labels_)
    # print(k_means.predict([[0, 0], [4, 4]]))
    # print(k_means.centroid)
