from cnn.parser import Parser
from miner.build import Measure, Model, Ranker, VectorSpace

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
            output.write("\n"*3)
            for method in [Measure.EUCLIDEAN, Measure.COSINE, Measure.JACCARD]:
                print("Method: {}\n".format(method.name), file=output)
                print("Pair\tScore\n",file=output)
                ds = Ranker.pairwise_distances(model.matrix,method)
                print("\n".join(
                    "{}\t{}".format(*pair) for pair in Ranker.rank(ds)
                ), file=output)
                output.write("\n"*3)

    return 0


if __name__ == "__main__":
    main()
