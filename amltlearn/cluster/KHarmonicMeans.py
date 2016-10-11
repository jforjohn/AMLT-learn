"""K-Harmonic Means clustering"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import as_float_array
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.sparsefuncs import mean_variance_axis


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
        self.labels_ = None
        self.n_iter = None
        self.performance = None

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=[np.float64,
                                                       np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
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

        # Check if the model had fitted the data previously
        if self.X != X:
            self.X = X

            self.centroid, self.labels_, self.performance,\
            self.n_iter = k_harmonic_means(
                X, n_clusters=self.n_clusters, n_init=self.n_init,
                max_iter=self.max_iter, verbose=self.verbose,
                tol=tol, random_state=random_state,
                n_jobs=self.n_jobs)

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

        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)

        if X == self.X:
            res = self.labels_
        else :
            x_squared_norm = row_norms(X, squared=True)
            L2_distance = np.max(euclidean_distances(
                X=X, Y=self.cluster_centers_, squared=False,
                X_norm_squared=x_squared_norm,
                Y_norm_squared=self.y_squared_norms_), self.e)

            L2_p2_distance = L2_distance ** (self.p+2)
            res = _calc_membership(X, self.cluster_centers_, self.p,
                                   L2_p2_distance=L2_p2_distance,
                                   e=self.e)

        return res


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

    best_labels = None
    best_inertia = None
    best_centers = None
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
            membership, inertia, centroid, n_iter = \
                kmeans_single(X, n_clusters, max_iter=max_iter,
                              verbose=verbose,
                              x_squared_norm=x_squared_norm,
                              tol=tol, random_state=random_state,
                              p=p, e=e)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = membership.copy()
                best_centers = centroid.copy()
                best_inertia = inertia
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
        membership, inertia, centroid, n_iter = zip(*results)
        best = np.argmin(inertia)
        best_labels = membership[best]
        best_inertia = inertia[best]
        best_centers = centroid[best]
        best_n_iter = n_iter[best]

    if not sp.issparse(X):
        best_centers += X_mean

    return best_centers, best_labels, best_inertia, best_n_iter


def _k_harmonic_means(X, n_clusters, max_iter=300, verbose=False,
                      x_squared_norm=None, tol=1e-4,
                      random_state=None, p=3.5, e=1e-8):
    """K-harmonic means clustering algorithm.


    Parameters
    ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

    """

    if not x_squared_norm:
        x_squared_norm = row_norms(X, squared=True)

    random_state = check_random_state(random_state)

    best_labels = None
    best_inertia = None
    best_centers = None

    # init
    centroid = _init_centroids(X, n_clusters,
                               random_state=random_state)

    # Allocate memory to store the partial results in order to speed
    # up the overall computation cost and pre-calculations
    membership = np.zeros(shape=(X.shape[0], n_clusters),
                          dtype=float)
    weight = membership.copy()
    # pre-calculations
    y_squared_norm = row_norms(centroid, squared=True)
    # shape (n_instances, n_centroids)
    L2_distance = np.max(
        euclidean_distances(X=X, Y=centroid,
                            Y_norm_squared=y_squared_norm,
                            squared=False,
                            X_norm_squared=x_squared_norm), e)
    inv_L2_p_dist = 1 / (L2_distance ** p)
    inv_L2_p2_dist = 1 / (L2_distance ** (p + 2))
    performance = _evaluate_performance(X, centroid, inv_L2_p_dist,
                                        p, e)

    if verbose:
        print('Initialization complete, with initial performance: '
              ''.join(str(performance)))

    # iterations
    iter_ = 0
    performance_incremment = np.inf
    while iter_ < max_iter and performance_incremment > tol:
        # Save state
        centroid_old = centroid.copy()
        performance_old = performance

        # Pre-calculations
        sum_dist_p2 = np.sum(inv_L2_p2_dist, axis=1)
        sum_dist_p = (np.sum(inv_L2_p_dist, axis=1)) ** 2
        membership = np.divide(inv_L2_p2_dist, sum_dist_p2)
        div_weight = np.divide(sum_dist_p2, sum_dist_p)
        m_k_i_numerator = np.multiply(membership, div_weight)
        m_k_i_denominator = np.sum(m_k_i_numerator, axis=0)
        m_k_i_numerator = np.multiply(X, m_k_i_numerator)
        mk_numerator = np.sum(m_k_i_numerator, axis=0)
        m_k_denominator = np.sum(m_k_i_denominator, axis=0)
        m_k = np.divide(mk_numerator, m_k_denominator)
        # m_k_i = np.divide(np.sum(np.multiply(X,m_k_i_numerator),
        #                          axis=0), np.sum(m_k_i_denominator),
        #                   0)

        print('Miraculous of lenght: '.join(str(m_k.shape[0])))


        # Calculate distances
        y_squared_norm = row_norms(centroid_old, squared=True)
        y_norm = np.sqrt(y_squared_norm)
        L2_distance = np.max(euclidean_distances(
            X=X, Y=centroid, squared=False), e)
        inv_L2_p_dist = 1 / (L2_distance ** (p))
        inv_L2_p2_dist = 1 / (L2_distance ** (p + 2))

        performance = _evaluate_performance(X, centroid)
        performance_incremment = performance_old - performance

        print('Performance: '.join(str(performance)))

        iter_ += 1

    # for i in range(n_samples):
    #     denominator = np.sum(distance[i, :])
    #     for j in range(k):
    #         membership[i, j] = distance[i, j] / denominator
    #
    #     # labels assignment is also called the E-step of EM
    #     labels, inertia = _labels_inertia(X, x_squared_norms,
    # centers,
    #                         precompute_distances=precompute_distances,
    #                         distances=distances)
    #
    #
    #
    # while difference_performance > error_threshold & & iteration <
    #     max_num_iterations
    #
    #     old_performance = performance;
    #     centroids = getNewCentroidsHKM(instances, centroids);
    #     performance = evaluatePerformance(instances, centroids);
    #     difference_performance = abs(performance - old_performance);
    #
    #     iteration = iteration + 1;
    # end

    # membership = np.sum(div_membership)
    # membership = div_membership
    n_iter = iter_

    return centroid, membership, n_iter, performance


def _evaluate_performance(X, centroid, inv_distance=None, p=3.5,
                          e=1e-8):
    """Evaluate performance of k-harmonic means clustering result.

    Parameters
    ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

    """

    if not inv_distance:
        # shape (n_instances, n_centroids)
        L2_distance = np.max(euclidean_distances(X=X, Y=centroid,
                                                 squared=False), e)
        inv_distance = 1 / (L2_distance ** p)

    performance = centroid.shape[0] * np.sum(1 / (np.sum(inv_distance,
                                                         axis=1)))
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

    if sp.issparse(centers):
        print("line 339: it was sparse")
        centers = centers.toarray()
    else:
        print("line 339: it was NOT sparse")

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
    membership = np.zeros(shape=(n_samples,k),
                          dtype=X.dtype)
    if L2_p2_distance is None:
        L2_distance = np.max(euclidean_distances(
            X=X, Y=centroid, squared=False), e)
        L2_p2_distance = L2_distance ** (p + 2)

    distance = 1/L2_p2_distance
    for i in range(n_samples):
        denominator = np.sum(distance[i, :])
        for j in range(k):
            membership[i, j] = distance[i, j] / denominator

    return membership

if __name__ == "__main__":
    print('Start')
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    k_means = KHarmonicMeans(n_clusters=2, random_state=0).fit(X)
    print('Finish')
    print(k_means.labels_)
    print(k_means.predict([[0, 0], [4, 4]]))
    print(k_means.centroid)
