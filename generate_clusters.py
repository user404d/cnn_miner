from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import nltk.cluster.kmeans as kmeans
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cnn.parser import Parser
from miner.build import Measure, Model, Ranker, VectorSpace


def plot_embedding(X, y, labels, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], labels[i][:2],
                 color=plt.cm.Set1(y[i] / 8.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)


def plot_sse_data(sse_data):
    plt.figure()
    ax = plt.subplot(111)
    ax.boxplot(sse_data)
    ax.set_title("Comparison of Distance Functions")
    ax.set_ylabel("Sum Squared Error")
    ax.set_xlabel("Distance Function")
    ax.set_xticklabels(["Euclidean", "Cosine", "Jaccard"])


def euclidean_distance(a, b):
    return 1 - Measure.euclidean(np.asmatrix(a), np.asmatrix(b))


def cosine_distance(a, b):
    return 1 - Measure.cosine(np.asmatrix(a), np.asmatrix(b))


def jaccard_distance(a, b):
    return 1 - Measure.jaccard(np.asmatrix(a), np.asmatrix(b))


def sse(a, center):
    return np.sum(np.square(a - center))


def clusters():
    with open("meta/categorized_stories.csv") as lines:
        stories = {}
        base_classifications = {}
        for line in lines:
            klass, path = line.strip().split(",")
            base_classifications[path] = klass
            with open(path) as story:
                stories[path] = Parser.tokenize(story.read())

    whole = Model(stories, VectorSpace.NORMALIZED)

    vector_space = [np.ravel(whole.matrix[i, ])
                    for i in range(len(whole.matrix))]

    csv_data = []
    titles = ["Euclidean", "Cosine", "Jaccard"]
    tsne_pca = TSNE(n_components=2, init='pca', random_state=0)
    reduced_matrix = tsne_pca.fit_transform(whole.matrix)
    categorized = [base_classifications[label] for label in whole.labels]
    for i, distance in enumerate([euclidean_distance, cosine_distance, jaccard_distance]):
        model = kmeans.KMeansClusterer(
            8, distance, repeats=4, avoid_empty_clusters=True)
        model.cluster(vector_space)
        means = model.means()
        classified = [model.classify(vector) for vector in vector_space]
        plot_embedding(reduced_matrix, classified, categorized,
                       "t-SNE K Means using {} Distance".format(titles[i]))
        csv_data.append(
            [sse(vector_space[i], means[j]) for i, j in enumerate(classified)])

    print("\n".join("{},{},{}".format(*triple)
                    for triple in zip(csv_data[0], csv_data[1], csv_data[2])))
    plot_sse_data(csv_data)
    plt.show()


if __name__ == "__main__":
    clusters()
