"""K-Harmonic Means clustering"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import as_float_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


class KHarmonicMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-harmonic means Clustering Algorithm

    The k-harmonic means algorithm (KHMp) follows the same strategy
    that k-means [1]_, differing in the objective function [2]_.

    Parameters
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
    algorithm : Feature NOT yet implemented
        K-harmonic means algorithm to use (e.g. {'Zhang',}).
    precompute_distances : Feature NOT yet implemented
        Pre-compute distances (faster but takes more memory).
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare
        convergence
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.
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

    Attributes
    ----------
    centroid : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    performance : float
        Sum of distances of samples to their closest cluster center.

    See also
    --------
    KMeans
        The classic implementation of the clustering method based on
        the Lloyd's algorithm. It consumes the whole set of input data
        at each iteration.
    FuzzyKMeans
        Different adaptation of k-means algorithm.

    Notes
    -----
    This version of K-harmonic means is the one presented by Zhang in
    [2]_, which is referred as the Generalized K-Harmonic Means,
    or KHMp, since the distance measure considered is the pth power of
    the L2-distance (Euclidean), instead of the squared L2 as in the
    previous versions of the k-harmonic means algorithm.
    Distance function d(x,m), implemented for KHMp (see [2]_):
    .. math:: max(||x_i - c_j||, e) ** 2,
    where the parameter e is to avoid denominators of value=0.
    The method was implemented taking scikit-learn KMeans as a guiding
    reference, moreover, it can be noticed that most of parameters and
    functions structural strategies have been maintained. However,
    some functionalities, such as different initialization methods for
    the centroids, which is crucial in k-means, while being redundant
    to KHM, since this one, is insensible to centroids
    initializations.
    In the performance formula implemented for KHMp, the parameter p
    default value was set to 3.5, since Zhang, in [2]_, stated that
    3.5 was the best he found after several experiments with different
    configurations. However, notice that he also stated that for high
    dimensionality data (i.e. > 2 dimensional data), larger p values
    could be desired.

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
    >>> from amltlearn.cluster import KHarmonicMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> kmeans = KHarmonicMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])

    """

    def __init__(self, n_clusters=8, n_init=10, max_iter=300,
                 tol=1e-4, p=3.5, e=1e-8, precompute_distances='True',
                 verbose=False, random_state=None, n_jobs=1,
                 algorithm='Zhang'):

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
        """Verify that the number of samples given is larger than
        k, and the data consistency for the clustering process

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

        Return
        ------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data to be clustered checked and fixed.

        """

        X = check_array(X, accept_sparse='csr', dtype=[np.float64,
                                                       np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        """Check consistency of data in array, in order to be
        clustered properly

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data to be clustered.

        Return
        ------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data to be clustered checked and fixed.

        """

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
            Coordinates of the data points to cluster.

        Return
        ------
        model
            Clustering model built on the data represented in X,
            ready for further clustering new data on the built
            clusters.

        """

        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        tol = _tolerance(X, self.tol)

        self.centroid, self.membership, self.performance, self.n_iter\
            = k_harmonic_means(X, n_clusters=self.n_clusters,
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
            Coordinates of the data points to cluster.

        Return
        ------
        labels : array-like, int, shape [n_samples,]
            List of indexes of the clusters where each data
            sample belongs to.

        """

        return self.fit(X).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `centroid` is
        called the code book and each value returned by `predict` is
        the index of the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples,
        n_features)
            New data to predict.

        Returns
        -------
        labels : array, int, shape [n_samples,]
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
    """Show cluster with higher membership of each sample

    Parameters
    ----------
    membership : matrix, float
        Membership map of each sample with each cluster.

    Returns
    -------
    labels : array, int, shape [n_samples,]
        Index of the cluster each sample belongs to.

    """

    return np.argmax(membership, axis=1)


def _tolerance(X, tol):
    """Return a tolerance which is independent of the data set

    Parameters
    ----------
    tol : float
        Relative tolerance with regards to inertia to declare
        convergence

    """

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
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    tol : float, optional
        The relative increment in the results before declaring
        convergence.
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.
    verbose : boolean, optional
        Verbosity mode.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    n_jobs : int
        The number of jobs to use for the computation. This works by
        computing each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing
        code is used at all, which is useful for debugging. For n_jobs
        below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used.
    algorithm : Feature NOT yet implemented feature.
    precompute_distances : Feature NOT yet implemented feature.

    Returns
    -------
    best_centroid : float matrix, shape=(n_clusters, n_features)
        Centroids found at the last iteration of k-harmonic means,
        corresponding to the run which retrieved 'best results', this
        is determined by the run that retrieved lower final
        'performance' value.
    best_membership: float matrix, shape=(n_samples, n_clusters)
        membership[i,:] shows the membership degree of the i'th
        observation to each cluster, corresponding to the run which
        retrieved 'best results', this is determined by the run
        that retrieved lower final 'performance' value.
    best_performance: float
        The final value of the performance criterion (see [2]_),
        corresponding to the run which
        retrieved 'best results', this is determined by the run
        that retrieved lower final 'performance' value.
    best_n_iter: int
        Number of iterations, corresponding to the run which
        retrieved 'best results', this is determined by the run
        that retrieved lower final 'performance' value.

    References
    ----------
    .. [2] B. Zhang. Generalized k-harmonic means – boosting in
    unsupervised learning. Technical Report HPL-2000-137,
    Hewlett-Packard Labs, 2000.

    """

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero."
                         % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive '
                         'number, got %d instead' % max_iter)

    X = as_float_array(X, copy=True)

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

    # pre-compute squared norms of the data samples
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
            if verbose:
                print('Run #'+str(it) + '/' + str(n_init)
                      + ' completed\n  Performance: '
                      + str(performance) + '\n  #Iterations: '
                      + str(n_iter))
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

    if verbose:
        print('Clustering completed\nBest results achieved:\n  '
              'Performance: ' + str(best_performance)
              + '\n  #Iterations: ' + str(best_n_iter))

    return best_centroid, best_membership, best_performance, \
        best_n_iter


def _k_harmonic_means(X, n_clusters, max_iter=300, verbose=False,
                      x_squared_norm=None, tol=1e-4,
                      random_state=None, p=3.5, e=1e-8):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters: int
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    tol: float, optional
        The relative increment in the results before declaring
        convergence.
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.
    verbose: boolean, optional
        Verbosity mode
    x_squared_norm: array
        Precomputed x_squared_norms.
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    -------
    centroid: float matrix, shape=(n_clusters, n_features)
        Centroids found at the last iteration of k-harmonic means.
    membership: float matrix, shape=(n_samples, n_clusters)
        membership[i,:] shows the membership degree of the i'th
        observation to each cluster.
    performance: float
        The final value of the performance criterion (see [2]_).
    n_iter : int
        Number of iterations in this run.

    References
    ----------
    .. [2] B. Zhang. Generalized k-harmonic means – boosting in
    unsupervised learning. Technical Report HPL-2000-137,
    Hewlett-Packard Labs, 2000.

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
        print('Initialization complete')

    # iterations
    iter_ = 0
    performance_increment = np.inf
    while iter_ < max_iter and performance_increment > tol:
        # Save previous state
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

        # Calculate performance
        performance = _evaluate_performance(X, centroid,
                                            inv_L2_p_dist, p, e)
        performance_increment = performance_old - performance
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
    X : array-like or matrix, shape (n_samples, n_features)
        The observations to cluster.
    centroid : float ndarray with shape (k, n_features)
        Centroids found when computing K-harmonic means.
    p : int, default: 3.5
        Power of the KHMp's distance formula, which is:
        .. math :: d(x, m) = (L2-distance)^p
    inv_distance : array-like or matrix, shape (n_instances,
    n_centroids), float, default : None
        Precomputed inverted-distances for speeding up the membership
        calculations.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.

    Return
    ------
    performance : float
        Quality measure of the clustering partition.

    """

    if inv_distance is None:
        # shape (n_instances, n_centroids)
        inv_distance = 1 / ((np.maximum(euclidean_distances(
            X=X, Y=centroid, squared=False), e)) ** p)

    performance = np.sum(centroid.shape[0] / (np.sum(inv_distance,
                                                     axis=1)))
    return performance


def _init_centroids(X, n_centroid, random_state=None):
    """Compute the initial centroids.

    Random initialization of the centroids.

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
    n_centroid: int
        number of centroids
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    -------
    centroid: array, shape(k, n_features)

    """

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if n_samples < n_centroid:
        raise ValueError("n_samples=%d should be larger than k=%d" %
                         (n_samples, n_centroid))

    seeds = random_state.permutation(n_samples)[:n_centroid]
    centroid = X[seeds]
    _validate_center_shape(X, n_centroid, centroid)
    return centroid


def _validate_center_shape(X, n_centers, centroid):
    """Check if centers is compatible with X and n_centers"""
    if len(centroid) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centroid.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centroid.shape[1], X.shape[1]))


def _calc_membership(X, centroid, p=3.5, inv_L2_p2_distance=None,
                     e=1e-8):
    """Calculate the memberships of instances to existing clusters

    Parameters
    ----------
    X : array-like or matrix, shape (n_samples, n_features)
        The observations to cluster.
    centroid : float ndarray with shape (k, n_features)
        Centroids found when computing K-harmonic means.
    p : int, default: 3.5
        Power of the L^2 distance as the d(x,m) in KHMp.
    inv_L2_p2_distance : array-like or matrix, shape (n_instances,
    n_centroids), float, default : None
        Precomputed inverted-distances for speeding up the membership
        calculations.
    e : float, default: 1e-8
        Small positive value necessary for avoiding zero denominators.

    Return
    ------
    membership : float matrix with shape (n_samples, k)

    """

    if inv_L2_p2_distance is None:
        inv_L2_p2_distance = 1 / ((np.maximum(euclidean_distances(
            X=X, Y=centroid, squared=False), e)) ** (p + 2))

    sum_dist_p2 = np.array([np.sum(inv_L2_p2_distance, axis=1)]).T
    membership = np.divide(inv_L2_p2_distance, sum_dist_p2)
    return membership


if __name__ == "__main__":
    """Testing the clustering algorithm on the iris data set"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.cluster import KMeans
    import sklearn.metrics as sm
    from sklearn import datasets
    import warnings

    # It is interesting to suppress warnings since there is an
    # internal issue with the 'euclidean_distances' sklearn.metrics
    # function, when it tries to check the precomputed squared
    # norms, the array checking function raises a warning of
    # deprecated method for 1-d arrays. Since most times the
    # precomputed norms are used for speeding up the code, in this
    # test the warnings are suppressed, for increasing readability
    # on the results.
    warnings.filterwarnings("ignore")

    print('Starting the integrated simple test of K-harmonic means '
          'clustering algorithm, implemented for the AMLT subject.'
          '\n\nThis test will run the mentioned algorithm '
          'for classifying the iris data set in 3 clusters; the '
          'K-means standard method, with the equivalent '
          'configuration; and finally will show 3 plots:\n    '
          '1-KHarmonic clustering result\n    2-KMeans clustering '
          'result\n    3-Ground truth of the iris data.')

    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Add further clustering algorithms or configurations as
    # elements of the 'estimators' list, in order to compare the
    # their performance, by analysing the retrieved results on the
    # well-known iris data set. Be aware that this simple test is
    # prepared for plotting just 3 clusters, if more are desired
    # take into account that the labels might be in a different
    # order and this code wont find it, though it should be done
    # manually.
    estimators = {'KHarmonicMeans': KHarmonicMeans(n_clusters=3),
                  'KMeans': KMeans(n_clusters=3)}

    best_order = None
    fignum = 1
    for name, est in estimators.items():
        print('\nClustering the iris data set by the ' + str(name)
              + ' clustering algorithm')
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        est.fit(X)
        labels = est.labels_

        best_predict_y = labels
        best_accuracy = 0
        best_order = [0, 1, 2]
        orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0],
                  [2, 0, 1], [2, 1, 0]]
        for order in orders:
            predict_y = np.choose(est.labels_, order).astype(np.int64)
            accuracy = sm.accuracy_score(y, predict_y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predict_y = predict_y
                best_order = order

        print('Accuracy: ' + str(sm.accuracy_score(y,
                                                   best_predict_y)))
        print('Confusion Matrix: ')
        print(sm.confusion_matrix(y, best_predict_y))

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=best_predict_y.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        fignum += 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [0, 1, 2]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')

    print('\nEnd of the simple test.')
    plt.show()

# EOF
