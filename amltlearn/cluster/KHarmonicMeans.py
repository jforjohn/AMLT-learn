"""K-Harmonic Means clustering"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


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
        Maximum number of iterations of the k-harmonic means algorithm
        for a single run.
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
        original data is not modified. If False, the original data is modified, and put back before the function returns, but small
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
    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.
    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)
    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.
    Distance function d(x,m), implemented for KHMp (see [2]_):
    .. math:: max(||x_i - c_j||, e)
    where the parameter e is to avoid zero denominators.
    Performance function implemented for KHMp was presented in [2]_.
    The paramater p default value was set to 3.5, as that was the best value found by Zhang [2]_, who also stated that for high
    dimensionality data ( > 2 dimensions), larger p values could be
    desired.

    References
    ----------
    .. [1] J. MacQuenn. Some methods for classification and analysis of
    multivariate observations. In L. M. LeCam and J. Neyman, editors,
    Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics
    and Probability, volume 1, pages 281-297, Berkeley, CA, 1967. University
    of California Press.
    .. [2] B. Zhang. Generalized k-harmonic means – boosting in unsupervised
    learning. Technical Report HPL-2000-137, Hewlett-Packard Labs, 2000.

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
                 max_iter=300, tol=1e-4, p=3.5, e=1e-8, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.p = p
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.e = e

    ###########################
    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, y=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
        return self
    
    ############################

    def fit(self, X):
        """Clusters the examples

        :param X:
        :return:
        """

        if self.algorithm == 'classical':
            self.cluster_centers_, self.labels_, self.inertia_ = \
                self._fit_process(X)
        # elif self.algorithm == 'bagirov':
            # self.cluster_centers_, self.labels_, self.inertia_ = \
                # self._fit_process_bagirov(
                    # X)

        return self

    def predict(self, X):
        """
        Returns the nearest cluster for a data matrix

        @param X:
        @return:
        """
        clasif = []
        for i in range(X.shape[0]):
            ncl, mdist = self._find_nearest_cluster(X[i].reshape(1, -1),
                                                    self.cluster_centers_)
            if mdist <= self.radius:
                clasif.append(ncl)
            else:
                clasif.append(-1)
        return clasif

    def _fit_process(self, X):
        """
        Classical global k-means algorithm

        :param X:
        :return:
        """

        # Compute the centroid of the dataset
        centroids = sum(X) / X.shape[0]
        centroids.shape = (1, X.shape[1])

        for i in range(2, self.n_clusters + 1):
            mininertia = np.infty
            for j in range(X.shape[0]):
                newcentroids = np.vstack((centroids, X[j]))
                # print newcentroids.shape
                km = KMeans(n_clusters=i, init=newcentroids, n_init=1)
                km.fit(X)
                if mininertia > km.inertia_:
                    mininertia = km.inertia_
                    bestkm = km
            centroids = bestkm.cluster_centers_

        return bestkm.cluster_centers_, bestkm.labels_, bestkm.inertia_

    def _fit_process_bagirov(self, X):
        """
        Clusters using the global K-means algorithm Bagirov variation
        :param X:
        :return:
        """

        # Create a KNN structure for fast search
        self._neighbors = NearestNeighbors()
        self._neighbors.fit(X)

        # Compute the centroid of the dataset
        centroids = sum(X) / X.shape[0]
        assignments = [0 for i in range(X.shape[0])]

        centroids.shape = (1, X.shape[1])

        # compute the distance of the examples to the centroids
        mindist = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            mindist[i] = euclidean_distances(X[i].reshape(1, -1),
                                             centroids[assignments[i]].reshape(
                                                 1, -1), squared=True)[0]

        for k in range(2, self.n_clusters + 1):
            newCentroid = self._compute_next_centroid(X, centroids,
                                                      assignments, mindist)
            centroids = np.vstack((centroids, newCentroid))
            km = KMeans(n_clusters=k, init=centroids, n_init=1)
            km.fit(X)
            assignments = km.labels_
            for i in range(X.shape[0]):
                mindist[i] = euclidean_distances(X[i].reshape(1, -1),
                                                 centroids[
                                                     assignments[i]].reshape(1,
                                                                             -1),
                                                 squared=True)[0]

        return km.cluster_centers_, km.labels_, km.inertia_

    def _compute_next_centroid(self, X, centroids, assignments, mindist):
        """
        Computes the candidate for the next centroid

        :param X:
        :param centroids:
        :return:
        """
        minsum = np.infty
        candCentroid = None

        # Compute the first candidate to new centroid
        for i in range(X.shape[0]):
            distance = euclidean_distances(X[i].reshape(1, -1),
                                           centroids[assignments[i]].reshape(1,
                                                                             -1))[
                0]
            S2 = self._neighbors.radius_neighbors(X[i].reshape(1, -1),
                                                  radius=distance,
                                                  return_distance=False)[0]
            S2centroid = np.sum(X[S2], axis=0) / len(S2)
            S2centroid.shape = (1, X.shape[1])
            cost = self._compute_fk(X, mindist, S2centroid)

            if cost < minsum:
                minsum = cost
                candCentroid = S2centroid

        # Compute examples for the new centroid
        S2 = []
        newDist = euclidean_distances(X, candCentroid.reshape(1, -1),
                                      squared=True)
        for i in range(X.shape[0]):
            if newDist[i] < mindist[i]:
                S2.append(i)

        newCentroid = sum(X[S2]) / len(S2)
        newCentroid.shape = (1, X.shape[1])

        while not (candCentroid == newCentroid).all():
            candCentroid = newCentroid
            S2 = []
            newDist = euclidean_distances(X, candCentroid.reshape(1, -1),
                                          squared=True)
            for i in range(X.shape[0]):
                if newDist[i] < mindist[i]:
                    S2.append(i)

            newCentroid = np.sum(X[S2], axis=0) / len(S2)
            newCentroid.shape = (1, X.shape[1])

        return candCentroid

    def _compute_fk(self, X, mindist, ccentroid):
        """
        Computes the cost function

        :param X:
        :param mindist:
        :param ccentroid:
        :return:
        """

        # Distances among the examples and the candidate centroid
        centdist = euclidean_distances(X, ccentroid.reshape(1, -1),
                                       squared=True)

        fk = 0
        for i in range(X.shape[0]):
            fk = fk + min(mindist[i], centdist[i][0])

        return fk

    def _harmonic_distance(self, X, C):
        """Calculate the harmonic distance between two points
        
        Parameters
        ----------
        X : array_like
            Data instance
        C : array_like
            Center coordinates
        
        Return
        ------
        dist : float
            Harmonic distance
        """
        dist = max(sqrt(sum((X-C).^2)), self.e)
        return dist
        
    @staticmethod
    def _find_nearest_cluster(examp, centers):
        """
        Finds the nearest cluster for an example
        :param examp:
        :param centers:
        :return:
        """

        dist = _harmonic_distance(centers, examp.reshape(1, -1))
        # dist = euclidean_distances(centers, examp.reshape(1, -1))

        pmin = np.argmin(dist)
        vmin = np.min(dist)

        return pmin, vmin
