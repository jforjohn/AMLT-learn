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
    The negentropy increment is calculated as in [1]_:
    .. math :: i_neg = 0.5 * (sum(p_i * log(cov(X_i))) - log(cov(X)))
                       - sum(p_i * log(p_i))

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
        single_cov_det_log = np.log(np.linalg.det(np.cov(X,
                                                  rowvar=False)))

    unique_label = le.classes_
    n_samples_per_label = bincount(label_,
                                   minlength=len(unique_label))
    order = np.argsort(label_)
    space_index = n_samples_per_label.cumsum()[:-1]
    cluster_space = np.array(np.split(X[order],
                             indices_or_sections=space_index,
                             axis=0))

    prior_proba = n_samples_per_label / n_sample
    # For each cluster elements matrix, compute the logarithm of
    # the determinant of the covariance matrix, and multiply it by
    # it's prior
    cluster_cov_det_log = np.sum([np.log(np.linalg.det(np.cov(
        cluster_space[cluster_it], rowvar=False))) for cluster_it
                                 in range(cluster_space.shape[0])]
                                 * prior_proba)

    prior_proba_log = np.sum(np.log(prior_proba) * prior_proba)
    # Calculate negentropy increment
    increment = (0.5 * (cluster_cov_det_log - single_cov_det_log)
                 - prior_proba_log)

    return increment


def _check_number_of_labels(n_labels, n_samples):
    """Check the consistence of the number of elements.

    Parameters
    ----------
    n_labels, n_samples : int
        Number of elements to be equal

    """

    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are "
                         "2 to n_samples - 1 (inclusive)" % n_labels)


if __name__ == "__main__":
    """Testing the negentropy increment clustering validation
    algorithm on generated data sets"""
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.utils import check_random_state
    import matplotlib.pyplot as plt

    # The test provided compares the resulting performance of
    # K-means when using different number of clusters on a generated
    # data set. It is expected that the lowest (best) negentropy
    # increment values will be retrieved when the numbers of clusters
    # used, are similar to the real number of clusters.

    print(
        'Starting the integrated simple test of Negentropy'
        'increment clustering validation metric (for crisp'
        'clustering algorithms).\nImplemented for the AMLT subject.'
        '\n\nFor this purpose, it will run different clustering'
        'algoritms/configurations,\n for clustering a generated data'
        'set, in order to check if the validation\n metric agrees'
        'with the real number of classes in the data set.')

    plt.figure(figsize=(12, 12))

    # Data set creation
    n_samples = 30000
    random_state = 170
    n_cluster_real = 5
    X, y = make_blobs(n_samples=n_samples, centers=n_cluster_real,
                      random_state=random_state)

    # Add further clustering algorithms or configurations as
    # elements of the 'estimators' list, in order to compare the
    # their performance, by analysing the retrieved results on the
    # generated data set, using the negentropy increment method.

    estimator = np.array([[('KMeans_' + str(n_cluster)), KMeans(
        n_clusters=n_cluster), 0., n_cluster] for n_cluster in range(2, 11)])

    # Pre-calculation
    single_cov_det_log = np.log(np.linalg.det(np.cov(X, rowvar=False)))
    for i in range(estimator.shape[0]):
        name = estimator[i, 0]
        est = estimator[i, 1]

        print('\nClustering the generated data set by the '
              + str(name) + ' clustering algorithm')

        label = est.fit_predict(X)
        # Plotting
        plt.subplot(3, 3, i+1)
        plt.scatter(X[:, 0], X[:, 1], c=label)
        plt.title(name)
        # Save results
        estimator[i, 2] = negentropy_cluster_increment(
            X, label, single_cov_det_log=single_cov_det_log)

    print('\nAll clustering process have finished.\n')

    # Print results
    sort_estimator = estimator[np.argsort(estimator[:, 2],
                                            axis=0)]
    print('\nNext the full classification from best to worse, of the'
          'compared clustering strategies, is shown:\n')
    additional_msg = ''
    for name, est, negentropy_increment, n_cluster in sort_estimator:
        if n_cluster_real == n_cluster:
            additional_msg = (
                '  => ' + str(n_cluster) + ' is the REAL NUMBER OF'
                                           'CLUSTERS IN THE DATASET')
        else:
            additional_msg = ''
        print('Clustering algorithm: ' + name
              + ', negentropy increment: '
              + str(negentropy_increment) + additional_msg)

    # Uncomment for visualizing the different partitions
    # plt.show()

    print('\nEnd of the simple test.')

    ##################################################################
    # ATTENTION
    # ---------
    # The following part of this section of the code, was just
    # implemented for the purpose of writing the report, though, it
    # should be removed before integrating the code with the rest of
    # the AMLT-learn repository.
    ##################################################################

    print('\nStart of the report test.')

    # The experiment results will be averaged over 10 runs
    n_run = 10

    # Initialisation
    random_state = 170
    random_state = check_random_state(random_state)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_run)
    neg_inc_result = {'Data1': np.zeros([10, 9]),
                      'Data2': np.zeros([10, 9]),
                      'Data3': np.zeros([10, 9]),
                      'Data4': np.zeros([10, 9])}
    estimator = np.array([KMeans(n_clusters=n_cluster) for n_cluster
                          in range(2, 11)])

    # Repetition loop
    i = 0
    for seed in seeds:
        # Data sets configuration
        dataset = {'Data1': make_blobs(n_samples=500, n_features=2,
                                       centers=4, cluster_std=1.,
                                       random_state=seed),
                   'Data2': make_blobs(n_samples=10000, n_features=2,
                                       centers=4, cluster_std=1.,
                                       random_state=seed),
                   'Data3': make_blobs(n_samples=3000, n_features=2,
                                       centers=2, cluster_std=1.,
                                       random_state=seed),
                   'Data4': make_blobs(n_samples=3000, n_features=2,
                                       centers=7, cluster_std=1.,
                                       random_state=seed)
                   }
        # For each data set
        for name, [X, y] in dataset.items():
            # Pre-calculation
            single_cov_det_log = np.log(
                np.linalg.det(np.cov(X, rowvar=False)))
            # Iterate over clustering configurations
            j = 0
            for est in estimator:
                pred_y = est.fit_predict(X)
                neg_inc_result[name][i, j] = \
                    negentropy_cluster_increment(
                        X, pred_y,
                        single_cov_det_log=single_cov_det_log)
                j += 1
        i += 1

    # print(neg_inc_result)
    # print(neg_inc_result.items())
    neg_final_result = {name: np.average(result, axis=0)
                        for name, result in neg_inc_result.items()}

    for name, result in neg_final_result.items():
        print(name)
        print(result)

    print('\nEnd of the report experiment.')

# EOF
