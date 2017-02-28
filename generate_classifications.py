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

def print_formatted_data(accs, fmeasures):
    print(" ,Accuracy," + ",".join("{:.2f}".format(acc) for acc in accs))
    print(" ,Average Accuracy,", sum(accs)/5)
    print(" ,Max Accuracy,", max(accs))
    print(" ,F-Measure," + ",".join("{:.2f}".format(fm) for fm in fmeasures))
    print(" ,Average F-Measure,", sum(fmeasures)/5)
    print(" ,Max F-Measure,", max(fmeasures))    

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
    KNN vs DTC
    """

    whole = Model(stories, VectorSpace.NORMALIZED)
    classifications = [base_classifications[label] for label in whole.labels]
    all_sims = Ranker.pairwise_distances(whole.matrix, Measure.JACCARD)

    knns = [
        KNN(n_neighbors=3, metric='precomputed', n_jobs=-1),
        KNN(metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=7,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=20,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=80,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=6,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=8,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=9,metric='precomputed', n_jobs=-1),
        KNN(n_neighbors=10,metric='precomputed', n_jobs=-1),
    ]
    dtcs = [DTC(), DTC(criterion='entropy')]

    knn_accs = [[]] * len(knns)
    knn_fmeasures = [[]] * len(knns)
    dtc_accs = [[]] * len(dtcs)
    dtc_fmeasures = [[]] * len(dtcs)

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
        dtc_classifiers = [ dtc.fit(x_train, y_train) for dtc in dtcs ]

        knn_predictions = [ 
            knn_classy.predict(ds_test) for knn_classy in knn_classifiers 
        ]

        dtc_predictions = [
            dtc_classy.predict(x_test) for dtc_classy in dtc_classifiers
        ]
        
        knn_accs = [
            accs + [knn_classy.score(ds_test,y_test)] for accs,knn_classy \
            in zip(knn_accs,knn_classifiers)
        ]

        dtc_accs = [
            accs + [dtc_classy.score(x_test,y_test)] for accs,dtc_classy \
            in zip(dtc_accs,dtc_classifiers)
        ]
        
        knn_fmeasures = [
            fmeasures + [f1(y_test, knn_prediction, average='weighted')] \
            for fmeasures,knn_prediction in zip(knn_fmeasures,knn_predictions)
        ]

        dtc_fmeasures = [
            fmeasures + [f1(y_test, dtc_prediction, average='weighted')] \
            for fmeasures,dtc_prediction in zip(dtc_fmeasures,dtc_predictions)
        ]

    """
    Plot stuff because Excel and the like are terrible...
    """
    inds = np.arange(5)
    w_inds = np.arange(0,5*2,2)
    ind_labels = tuple([str(i+1) for i in inds])
    colors = ('r','b','g','y','m','c','#0ef0ef0e','m','#efefefef')
    width = 0.35

    knn_accs_fig = plt.figure()
    knn_accs_bar = knn_accs_fig.add_subplot(111)
    knn_fmeasures_fig = plt.figure()
    knn_fmeasures_bar = knn_fmeasures_fig.add_subplot(111)
    for i,accs,fmeasures in zip(range(len(knns)), knn_accs, knn_fmeasures):
        if i < 5:
            accs_bar = knn_accs_bar.bar(w_inds+(width*i), accs, width, 
                                        color=colors[i], label="KNN {}".format(knns[i].n_neighbors))
            fmeasures_bar = knn_fmeasures_bar.bar(
                w_inds+(width*i), fmeasures, width, color=colors[i], label="KNN {}".format(knns[i].n_neighbors)
            )
        print("[*] KNN {}".format(knns[i].n_neighbors))
        print_formatted_data(accs, fmeasures)

    knn_accs_bar.set_title("KNN Accuracy per Fold")
    knn_accs_bar.set_xticks(w_inds+(width*2))
    knn_accs_bar.set_xticklabels(ind_labels)
    knn_accs_bar.set_yticks(np.arange(0, 1.0, .1))
    knn_accs_bar.set_xlabel("Fold")
    knn_accs_bar.set_ylabel("Accuracy")
    knn_accs_bar.legend()

    knn_fmeasures_bar.set_title("KNN F-Measure per Fold")
    knn_fmeasures_bar.set_xticks(w_inds+(width*2))
    knn_fmeasures_bar.set_xticklabels(ind_labels)
    knn_fmeasures_bar.set_yticks(np.arange(0, 1.0, .1))
    knn_fmeasures_bar.set_xlabel("Fold")
    knn_fmeasures_bar.set_ylabel("Weighted F-Measure")
    knn_fmeasures_bar.legend()


    dtc_accs_fig = plt.figure()
    dtc_accs_bar = dtc_accs_fig.add_subplot(111)
    dtc_fmeasures_fig = plt.figure()
    dtc_fmeasures_bar = dtc_fmeasures_fig.add_subplot(111)
    for i,accs,fmeasures in zip(range(len(dtcs)), dtc_accs, dtc_fmeasures):
        accs_bar = dtc_accs_bar.bar(inds+(width*i), accs, width, 
                                    color=colors[i], label="DTC {}".format(dtcs[i].criterion))
        fmeasures_bar = dtc_fmeasures_bar.bar(
            inds+(width*i), fmeasures, width, color=colors[i], label="DTC {}".format(dtcs[i].criterion)
        )
        print("[*] DTC {}".format(dtcs[i].criterion))
        print_formatted_data(accs, fmeasures)

    dtc_accs_bar.set_title("Decision Tree Classifier Accuracy per Fold")
    dtc_accs_bar.set_xticks(inds+(width/2))
    dtc_accs_bar.set_xticklabels(ind_labels)
    dtc_accs_bar.set_yticks(np.arange(0, 1.0, .1))
    dtc_accs_bar.set_xlabel("Fold")
    dtc_accs_bar.set_ylabel("Accuracy")
    dtc_accs_bar.legend()

    dtc_fmeasures_bar.set_title("Decision Tree Classifier F-Measure per Fold")
    dtc_fmeasures_bar.set_xticks(inds+(width/2))
    dtc_fmeasures_bar.set_xticklabels(ind_labels)
    dtc_fmeasures_bar.set_yticks(np.arange(0, 1.0, .1))
    dtc_fmeasures_bar.set_xlabel("Fold")
    dtc_fmeasures_bar.set_ylabel("Weighted F-Measure")
    dtc_fmeasures_bar.legend()

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
