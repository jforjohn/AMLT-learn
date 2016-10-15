"""Unsupervised evaluation metrics."""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 12/10/2016 21:44

import numpy as np

from sklearn.utils import check_X_y
from sklearn.utils.fixes import bincount
from sklearn.preprocessing import LabelEncoder


def negentropy_cluster_increment(X, label_, single_cov_det_log=None):
    """Negentropy Clustering Validation index.

    Calculate the 'negentropy increment' of the current clustering
    partition, relative to having all data in one single cluster.
    This index can be applied as a general tool to validate the
    outcome of any crisp clustering algorithm, and also to compare
    solutions provided by different algorithms for a single problem.


    Parameters
    ----------
    X : array-like or matrix, shape=(n_samples, n_features)
        Clustered observations.
    label_ : array-like, shape=(n_samples,)
        Clusters assigned to samples.
    single_cov_det_log : float, default : None
        Precomputed logarithm of the determinant of the covariance
        matrix of the single clustering partition, for all data.
        Interesting to be provided when wiling to compare several
        algorithms on the same data, for speeding up the
        calculations time.

    Return
    ------
    increment : float
        Negentropy increment obtained with the current clustering
        partition.

    See also
    --------
    Partition Negentropy Criterion (PNC)
        Similar validation index, but for 'fuzzy' clustering
        algorithms, instead of 'crisp'. Is not yet implemented into
        the AMLT-learn project, see [2] for further information on
        the PNC algorithm.

    Notes
    -----
    When evaluating the negentropy increment validation metric,
    notice that lower values indicate better clustering properties.
    Though, the selection rule used, should be, 'the lower - the
    better'.

    References
    ----------
    .. [1] Luis F. Lago-Fernández and Fernando Corbacho.
    Normality-based validation for crisp clustering. Pattern
    Recognition, 43, 3 {782 - 795}, 2010.
    .. [2] Luis F. Lago-Fernández and Fernando Corbacho.
    Fuzzy Cluster Validation Using the Partition Negentropy
    Criterion. "Artificial Neural Networks -- ICANN 2009: 19th
    International Conference, Limassol, Cyprus, September 14-17,
    2009, Proceedings, Part II", {235 - 244}, 2009.

    """

    # Check parameters consistency
    n_sample = X.shape[0]

    X, label_ = check_X_y(X, label_, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    # Ensure numerical indexes are used
    label_ = le.fit_transform(label_)
    n_label = len(le.classes_)
    _check_number_of_labels(n_label, n_sample)
    # Pre-calculate
    if single_cov_det_log is None:
        single_cov_det_log = np.log(
            np.linalg.det(np.cov(X, rowvar=False)))

    unique_label = le.classes_
    n_samples_per_label = bincount(label_,
                                   minlength=len(unique_label))
    order = np.argsort(label_)
    space_index = n_samples_per_label.cumsum()
    cluster_space = np.split(X[order],
                             indices_or_sections=space_index, axis=0)
    # Calculate negentropy increment
    prior_proba = n_samples_per_label / n_sample
    print(cluster_space)
    print(cluster_space.shape)
    cluster_cov_det_log = [np.log(np.linalg.det(np.cov(cluster))) for
                           cluster in cluster_space] * prior_proba
    prior_proba_log = np.log(prior_proba) * prior_proba

    increment = 0.5 * (np.sum(cluster_cov_det_log)
                       - single_cov_det_log) - np.sum(
        prior_proba_log)

    return increment


def _check_number_of_labels(n_labels, n_samples):
    """Check the consistence of the number of elements.

    Parameters
    ----------
    n_labels, n_samples : int
        Number of elements to be equal

    """

    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are"
                         "2 to n_samples - 1 (inclusive)" % n_labels)


if __name__ == "__main__":
    """Testing the negentropy increment clustering validation
    algorithm on the iris data set"""
    from sklearn.cluster import KMeans
    from sklearn import datasets
    # import warnings

    # It is interesting to suppress warnings since there is an
    # internal issue with the 'euclidean_distances' sklearn.metrics
    # function, when it tries to check the precomputed squared
    # norms, the array checking function raises a warning of
    # deprecated method for 1-d arrays. Since most times the
    # precomputed norms are used for speeding up the code, in this
    # test the warnings are suppressed, for increasing readability
    # on the results.
    # warnings.filterwarnings("ignore")

    print(
        'Starting the integrated simple test of Negentropy'
        'increment clustering validation metric (for crisp'
        'clustering algorithms), implemented for the AMLT subject.'
        '\n\nFor this purpose, it will run different clustering'
        'algoritms/configurations, for clustering the the iris data'
        'set, in order to check if the validation metric agrees'
        'with the real number of classes in the data set.')

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Add further clustering algorithms or configurations as
    # elements of the 'estimators' list, in order to compare the
    # their performance, by analysing the retrieved results on the
    # well-known iris data set, using the negentropy increment
    # method.
    #
    # The test provided compares the resulting performance of
    # K-means when using different number of clusters on the iris
    # data set. It is expected that the lowest (best) negentropy
    # increment values will be retrieved for numbers of clusters
    # close to the 3, since is the real number of classes in the
    # data.
    estimator = np.array([[('KMeans_' + str(n_cluster + 1)), KMeans(
        n_clusters=n_cluster + 1), 0.] for n_cluster in range(9)])

    # Pre-calculation
    single_cov_det_log = np.log(np.linalg.det(np.cov(X,
                                                     rowvar=False)))
    for name, est, negentropy_increment in estimator:
        print('\nClustering the iris data set by the ' + str(name)
              + ' clustering algorithm')
        print(type(est))
        est = est.fit(X)
        label = est.labels_
        negentropy_increment = negentropy_cluster_increment(
            X, label, single_cov_det_log)
    print('\nAll clustering process have finished.\n')

    # Print results
    sorted_estimator = np.sort(estimator, axis=2)
    print('Next the full classification from best to worse, of the'
          'compared clustering strategies, is shown:\n')
    for name, est, negentropy_increment in sorted_estimator:
        print('Clustering algorithm: ' + name
              + ', negentropy increment: '
              + str(negentropy_increment))

    print('\nEnd of the simple test.')

# EOF
