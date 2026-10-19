# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum


class Dataset(Enum):
    CMC = 'cmc'
    MGM = 'mgm'
    ADULT = 'adult'
    CAHOUSING = 'cahousing'
    INFORMS = 'informs'
    ITALIA = 'italia'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class AnonMethod(Enum):

    #Optimal Lattice Anonymization
    OLA = 'ola'

    # Top-Down Greedy
    TOPDOWN = 'topdown'

    # Top-Down Greedy with l-diversity
    TOPDOWN_LDIV = 'topdown_ldiv'

    # Cluster-based
    CLUSTER = 'cluster'

    # Datafly
    DATAFLY = 'datafly'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value

class ClassifierModel(Enum):
    SVM = 'svm'
    RF = 'rf'
    KNN = 'knn'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value