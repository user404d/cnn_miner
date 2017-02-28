"""
##
#  Build model
##
"""

#TODO: Break into components and separate into different files

import numpy as np
from enum import Enum
from itertools import chain
from operator import itemgetter


class Measure(Enum):
    EUCLIDEAN = 1
    COSINE = 2
    JACCARD = 3

    @staticmethod
    def euclidean(row, matrix):
        """
        Euclidean distance for vector in population.
        """
        euclidean = matrix - row
        euclidean = np.square(euclidean)
        ones = np.ones((matrix.shape[0],1), dtype=np.float64)
        return  ones / (ones + np.sqrt(np.sum(euclidean, axis=1)))

    @staticmethod
    def cosine(row, matrix):
        """
        Cosine similarity for vector in population.
        """
        dot_products = matrix * row.T
        measure = np.square(matrix).sum(axis=1) * np.square(row).sum(axis=1)
        return np.divide(dot_products, measure)

    @staticmethod
    def jaccard(row, matrix):
        """
        Jaccard similarity generalized to arbitrary magnitudes
        for vector in population.
        """
        jaccard_min_sums = np.minimum(matrix, row).sum(axis=1)
        jaccard_max_sums = np.maximum(matrix, row).sum(axis=1)
        return np.divide(jaccard_min_sums, jaccard_max_sums)


class VectorSpace(Enum):
    BOOLEAN = 1
    FREQUENCY = 2
    NORMALIZED = 3
    
    @staticmethod
    def existence(feature, features):
        """
        Tests if the feature exists in the feature set
        """
        return feature in features

    @staticmethod
    def frequency(feature, features):
        """
        Counts the occurrences of a feature in a feature set
        """
        return features.count(feature)


class Ranker:
    @staticmethod
    def pairwise_distances(matrix, method=Measure.EUCLIDEAN):
        """
        Computes the pairwise distance between each vector
        """
        size = range(matrix.shape[0])
        out = None
        if method == Measure.EUCLIDEAN:
            out = [Measure.euclidean(matrix[i,], matrix) for i in size]
        elif method == Measure.COSINE:
            out = [Measure.cosine(matrix[i,], matrix) for i in size]
        else:
            out = [Measure.jaccard(matrix[i,], matrix) for i in size]
        return np.triu(np.concatenate(out, axis=1), k=0)

    @staticmethod
    def rank(matrix):
        """
        Ranks the vectors based on the pairwise distances.

        Only takes values from the upper half of the triangular matrix as
        the values are duplicated in the lower half.
        """
        pairs = zip(*np.triu_indices(matrix.shape[0],1))
        all_pairs_distance = [(i, matrix[i]) for i in pairs]
        return sorted(all_pairs_distance, key=itemgetter(1), reverse=True)

    @staticmethod
    def split_distances(test, train, method=Measure.JACCARD):
        size = range(test.shape[0])
        out = None
        if method == Measure.EUCLIDEAN:
            out = [Measure.euclidean(test[i,], train) for i in size]
        elif method == Measure.COSINE:
            out = [Measure.cosine(test[i,], train) for i in size]
        else:
            out = [Measure.jaccard(test[i,], train) for i in size]
        return np.concatenate(out, axis=1).T


class Model:
    """
    model which will be used to draw comparisons between records/vectors
    (matrix: records x features)
    """

    def __init__(self, 
                 tokenized_articles, 
                 type_vector_space=VectorSpace.BOOLEAN):
        """
        Args:
          tokenized_articles :: list of articles that have had content tokenized
          type_vector_space :: (existence, frequency, normalized frequency)
        """
        if len(tokenized_articles) == 0:
            raise ValueError("No tokenized articles provided.\n\n{}"
                             .format(tokenized_articles))
        elif not type_vector_space in VectorSpace:
            raise ValueError("Not VectorSpace.\n\n{}".format(type_vector_space))

        self._type = type_vector_space
        self.labels = list(sorted(tokenized_articles.keys()))
        all_features = chain.from_iterable(tokenized_articles.values())
        features = {}
        for feature in all_features:
            features[feature] = features.get(feature, 0) + 1
        self.features = []
        for feature, count in features.items():
            isolated = 0
            for label in self.labels:
                if feature in tokenized_articles[label]: isolated += 1
            if isolated < len(self.labels) and isolated > 1 and count > 1:
                self.features.append(feature)
        self.features.sort()
        #TODO: make conversion function to change between representations
        self.matrix = []

        for label in self.labels:
            vector = list(range(len(self.features)))
            tokens = tokenized_articles[label]
            for i, feature in enumerate(self.features):
                if self._type == VectorSpace.BOOLEAN:
                    vector[i] = VectorSpace.existence(feature, tokens)
                else:
                    vector[i] = VectorSpace.frequency(feature, tokens)
            self.matrix.append(vector)

        if self._type == VectorSpace.BOOLEAN:
            self.matrix = np.asmatrix(self.matrix, np.float64)
        elif self._type == VectorSpace.FREQUENCY:
            self.matrix = np.asmatrix(self.matrix, np.float64)
        else:
            self.matrix = np.asmatrix(self.matrix, np.float64)
            self.matrix /= np.max(self.matrix, axis=0)
        

    def __str__(self):
        """
        String representation of the class
        (Note: will truncate features list if too large)
        """
        disp_labels = "\n".join("{},{}".format(i,label) for i,label in enumerate(self.labels))
        disp_features = ""
        if len(self.features) > 50:
            disp_features = "\n".join(self.features[:25] + [".\n.\n."]
                                      + self.features[-25:])
        else:
            disp_features = "\n".join(self.features)
        return ("\n{0}\nType: {2}\n{1}" \
                "\nLabels: {3}\n\n{4}" \
                "\n\n{1}\nFeatures: {5}\n\n{6}" \
                "\n\n{1}\nMatrix: {7}\n\n{8}" \
                "\n\n{1}\n{0}").format("***********************************", 
                                       "----------------", self._type,
                                       len(self.labels), disp_labels,
                                       len(self.features), disp_features,
                                       self.matrix.shape, self.matrix)
