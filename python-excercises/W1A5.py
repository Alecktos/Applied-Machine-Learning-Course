from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def one_point(neighbours):
    return map(lambda distance: 1/(distance + 0.01), neighbours)  # 1 divided by distance because we want far distance to give low weight


def calc_weight(dataset):
    return map(one_point, dataset)


def train_and_test_defined_part():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    print np.unique(target)  # print all flower types

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data[:-4], target[:-4])
    result = knn.predict(data[-4:])

    for key in range(len(result)):
        if result[key] != target[-4:][key]:
            print "index: " + str(key) + " fel: " + str(result[key]) + " skulle vara " + str(target[-4:][key])


def train_90_test_10():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    np.random.seed(0)
    indices = np.random.permutation(len(data)) #randomizes indices
    training_data = data[indices[:-10]]  # do not take last 10 from the randomized indices
    training_target = target[indices[:-10]]

    test_data = data[indices[-10:]] # take the last 10 from randmo indices to test on
    test_target = target[indices[-10:]]

    knn = KNeighborsClassifier(n_neighbors=9, weights=calc_weight)
    knn.fit(training_data, training_target)

    result = knn.predict(test_data)

    for key in range(len(test_data)):
        if result[key] != test_target[key]:
            print "index: " + str(key) + " fel: " + str(result[key]) + " skulle vara " + str(test_target[key])


def run_10_fold_crossvalidation():
    iris = datasets.load_iris()
    knn = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='kd_tree')
    predicted = cross_val_predict(knn, iris.data, iris.target, cv=10)
    score = metrics.accuracy_score(iris.target, predicted)
    print "score: " + str(score)

run_10_fold_crossvalidation()
# train_and_test_defined_part()
# train_90_test_10()

