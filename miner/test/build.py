from cnn.parser import Parser as p
from miner.build import Measure as mz
from miner.build import Model as m
from miner.build import Ranker as r
from miner.build import VectorSpace as v


paths = [
    "selected_stories/65ac6a35c36c1b10bf12bada19028f7db8605f76.story",
    "selected_stories/678817d5122216d20d3a12d3fa93a5dd82a5d6aa.story",
    "selected_stories/67b60e417abfee8f4e3466b514557ae0fdec077b.story",
    "selected_stories/6983688e3a9c4762a668e119c5f4659107646d7c.story",
    "selected_stories/69b696853a8b018139511db558782708fa6f5d35.story",
    "selected_stories/6a30b9d6ef29204d0e81c2153c29a8096174ca8a.story",
    "selected_stories/6bda7e9dbbb42f90edcc53d603e7ea6e5f17421a.story",
    "selected_stories/6d53469decf418cb26ea6e2c4b37a7a98163a6b8.story",
    "selected_stories/7015a2fff33d3b0c680e607ea2f0f7a06bbb8755.story",
    "selected_stories/72032f5ff79eb5d40bb567817e3847d7b1e41b5a.story",
    "selected_stories/755aad0f6b246a4c796478297e96764fbf695173.story",
    "selected_stories/78359963dd01a0470ff09d27d4ddd6eeb3422c1c.story",
    "selected_stories/7a2ae80b0997d7a5a4a7aee350e07dc4c1c197cb.story",
    "selected_stories/7e996bdffdfeadab016baa9cae4209706303ae68.story",
    "selected_stories/7f5fd7614f32586747f65545bebba418c3679d12.story",
    ]

if __name__ == "__main__":
    stories = {}

    for path in paths:
        with open(path) as cnn_story:
            stories[path] = [word.lower() for word in p.tokenize(cnn_story.read())]

    existence = m(stories, v.BOOLEAN)
    print(mz.euclidean(existence.matrix[0,], existence.matrix))
"""
    existence = m(stories, v.BOOLEAN)
    print(existence)
    print(mz.euclidean(existence.matrix[0,], existence.matrix))
    print(mz.cosine(existence.matrix[0,], existence.matrix))
    print(mz.jaccard(existence.matrix[0,], existence.matrix))
    to_be_ranked = r.pairwise_distances(existence.matrix, mz.EUCLIDEAN)
    print(to_be_ranked)
    print(r.rank(to_be_ranked))
    frequency = m(stories, v.FREQUENCY)
    print(frequency)
    print(mz.euclidean(frequency.matrix[0,], frequency.matrix))
    print(mz.cosine(frequency.matrix[0,], frequency.matrix))
    print(mz.jaccard(frequency.matrix[0,], frequency.matrix))
    print(r.pairwise_distances(frequency.matrix, mz.COSINE))
    normalized = m(stories, v.NORMALIZED)
    print(normalized)
    print(mz.euclidean(normalized.matrix[0,], normalized.matrix))
    print(mz.cosine(normalized.matrix[0,], normalized.matrix))
    print(mz.jaccard(normalized.matrix[0,], normalized.matrix))
    print(r.pairwise_distances(normalized.matrix, mz.JACCARD))
"""
