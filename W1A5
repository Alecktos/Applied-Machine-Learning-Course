from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


def one_point(neighbours):
    return map(lambda distance: 1/(distance + 0.01), neighbours)  # 1 divided by distance because we want far distance to give low weight


def calc_weight(dataset):
    return map(one_point, dataset)


def train_and_test_all():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    print np.unique(target)  # print all flower types

    knn = KNeighborsClassifier(n_neighbors=9, weights=calc_weight)
    knn.fit(data, target)
    result = knn.predict(data)

    for key in range(len(result)):
        if result[key] != target[key]:
            print "index: " + str(key) + " fel: " + str(result[key]) + " skulle vara " + str(target[key])


def train_90_test_10():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    np.random.seed(0)
    indices = np.random.permutation(len(data))
    training_data = data[indices[:-10]]  # do not take last 10
    training_target = target[indices[:-10]]

    test_data = data[indices[-10:]]
    test_target = target[indices[-10:]]

    knn = KNeighborsClassifier(n_neighbors=9, weights=calc_weight)
    knn.fit(training_data, training_target)

    result = knn.predict(test_data)

    for key in range(len(test_data)):
        if result[key] != test_target[key]:
            print "index: " + str(key) + " fel: " + str(result[key]) + " skulle vara " + str(test_target[key])


# train_and_test_all()
train_90_test_10()

