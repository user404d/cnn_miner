from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import nltk.cluster.kmeans as kmeans
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cnn.parser import Parser
from miner.build import Measure, Model, Ranker, VectorSpace


def euclidean_distance(a, b):
    return 1 - Measure.euclidean(np.asmatrix(a), np.asmatrix(b))


def cosine_distance(a, b):
    return 1 - Measure.cosine(np.asmatrix(a), np.asmatrix(b))


def jaccard_distance(a, b):
    return 1 - Measure.jaccard(np.asmatrix(a), np.asmatrix(b))


def sse(a, center):
    return np.sum(np.square(a - center))


def clusters():
    with open("meta/randomized_stories.txt") as lines:
        stories = {}
        for line in lines:
            path = line.strip()
            with open(path) as story:
                stories[path] = Parser.tokenize(story.read())

    whole = Model(stories, VectorSpace.FREQUENCY)
    whole.matrix = normalize(whole.matrix)

    vector_space = [np.ravel(whole.matrix[i, ])
                    for i in range(len(whole.matrix))]

    csv_data = []
    print("Euclidean,Cosine,Jaccard")

    for distance in [euclidean_distance, cosine_distance, jaccard_distance]:
        model = kmeans.KMeansClusterer(
            8, distance, repeats=4, avoid_empty_clusters=True)
        model.cluster(vector_space)
        means = model.means()
        classified = enumerate([model.classify(vector)
                                for vector in vector_space])
        classified = [sse(vector_space[i], means[j]) for i, j in classified]
        csv_data.append(classified)

    print("\n".join("{},{},{}".format(*triple)
                    for triple in zip(csv_data[0], csv_data[1], csv_data[2])))


if __name__ == "__main__":
    clusters()
