from cnn.parser import Parser
from miner.build import Measure, Model, Ranker, VectorSpace

def csv_format(row):
    return "{},{},{}".format(*row)

def feature_freqs(pair, model):
    first = model.matrix[pair[0],]
    second = model.matrix[pair[1],]
    return [(feature, first[0,i], second[0,i]) for i,feature in enumerate(model.features)]

def main():
    with open("cnn_stories_listing.txt") as paths:
        stories = {}
        for path in paths:
            with open(path.strip()) as story:
                stories[path] = Parser.tokenize(story.read())

    types = {
        "existence": Model(stories, VectorSpace.BOOLEAN),
        "frequency": Model(stories, VectorSpace.FREQUENCY),
        "normalized": Model(stories, VectorSpace.NORMALIZED)
    }

    for _type_, model in types.items():
        with open("output/{}.txt".format(_type_), "w+") as output:
            print("Beginning {}...".format(_type_))
            print(model, file=output)
            for method in [Measure.EUCLIDEAN, Measure.COSINE, Measure.JACCARD]:
                with open("output/{}_{}.csv".format(_type_,method), "w+") as dump_method, \
                     open("output/{}_{}_best_match.csv".format(_type_,method), "w+") as best:
                    # Output all pairs distance
                    print("Method: {}\n".format(method.name), file=dump_method)
                    print("Article_A,Article_B,Score",file=dump_method)
                    ds = Ranker.pairwise_distances(model.matrix,method)
                    ranks = Ranker.rank(ds)
                    print("\n".join(
                        "{0[0]},{0[1]},{1}".format(*pair) for pair in ranks
                    ), file=dump_method)
                    top = ranks[0][0]
                    # Output top pair and article vectors
                    print("Best Pair {}".format(top), file=best)
                    print("Term,Article_A,Article_B", file=best)
                    print("\n".join(
                        csv_format(term_freq) for term_freq in feature_freqs(top, model)
                    ), file=best)
    return 0


if __name__ == "__main__":
    main()
