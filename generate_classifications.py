from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as ttsplit, cross_val_score
from sklearn.metrics import f1_score as f1
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import chain
from cnn.parser import Parser
from miner.build import Measure, Model, Ranker, VectorSpace

def classifications():
    with open("meta/categorized_stories.csv") as lines:
        stories = {}
        base_classifications = {}
        for line in lines:
            klass, path = line.strip().split(",")
            base_classifications[path] = klass
            with open(path) as story:
                stories[path] = Parser.tokenize(story.read())

    """
    KNN testing
    """

    whole = Model(stories, VectorSpace.NORMALIZED)
    classifications = [base_classifications[label] for label in whole.labels]
    all_sims = Ranker.pairwise_distances(whole.matrix, Measure.JACCARD)

    knns = [
        KNN(n_neighbors=3, metric='precomputed', n_jobs=-1),
        KNN(metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=6,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=7,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=8,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=9,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=10,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=20,metric='precomputed', n_jobs=-1),
    ]
    dtc = DTC()

    knn_accs = [[]] * len(knns)
    knn_fmeasures = [[]] * len(knns)
    dtc_accs = []
    dtc_fmeasures = []

    for _ in range(5):
        x_train, x_test, y_train, y_test = ttsplit(
            whole.matrix, classifications, test_size=.2
        )

        test_indices = [ 
            np.where(np.all(whole.matrix==test,axis=1))[0][0] for test in x_test
        ]
        test_labels = [ whole.labels[index] for index in test_indices ]
        test_data = [
            (label, base_classifications[label]) for label in test_labels 
        ]
    
        ds = Ranker.pairwise_distances(x_train, Measure.JACCARD)
        ds = np.ones(ds.shape) - ds
        
        ds_test = Ranker.split_distances(x_test, x_train)
        ds_test = np.ones(ds_test.shape) - ds_test

        knn_classifiers = [ knn.fit(ds, y_train) for knn in knns ]
        dtc_classifier = dtc.fit(x_train, y_train)

        knn_predictions = [ 
            knn_classy.predict(ds_test) for knn_classy in knn_classifiers 
        ]

        dtc_prediction = dtc_classifier.predict(x_test)
        
        knn_accs = [
            accs + [knn_classy.score(ds_test,y_test)] for accs,knn_classy \
            in zip(knn_accs,knn_classifiers)
        ]

        dtc_accs.append(dtc_classifier.score(x_test, y_test))
        
        knn_fmeasures = [
            fmeasures + [f1(y_test, knn_prediction, average='weighted')] \
            for fmeasures,knn_prediction in zip(knn_fmeasures,knn_predictions)
        ]

        dtc_fmeasures.append(f1(y_test, dtc_prediction, average='weighted'))

    for i,accs,fmeasures in zip(range(len(knn_accs)), knn_accs, knn_fmeasures):
        print("[-] KNN {}".format(knns[i].n_neighbors))
        print("Accuracy," + ",".join("{}".format(acc) for acc in accs))
        print("Average Accuracy,", sum(accs)/5)
        print("Max Accuracy,", max(accs))
        print("F-Measure," + ",".join("{}".format(fm) for fm in fmeasures))
        print("Average F-Measure,", sum(fmeasures)/5)
        print("Max F-Measure,", max(fmeasures))

    print("[-] DTC")
    print("Accuracy," + ",".join("{}".format(acc) for acc in dtc_accs))
    print("Average Accuracy,", sum(dtc_accs)/5)
    print("Max Accuracy,", max(dtc_accs))
    print("F-Measure," + ",".join("{}".format(fm) for fm in dtc_fmeasures))
    print("Average F-Measure,", sum(dtc_fmeasures)/5)
    print("Max F-Measure,", max(dtc_fmeasures))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(all_sims, cmap=cmap)
    ax1.grid(True)
    plt.title('Heatmap of Similarities')
    fig.colorbar(cax, ticks=[0.25,0.5,0.75,1])
    plt.show()

if __name__ == "__main__":
    classifications()
