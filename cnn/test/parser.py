from cnn.parser import Parser as p
from functools import reduce

paths = [
    "cnn_stories/stories/0001d1afc246a7964130f43ae940af6bc6c57f01.story",
    "cnn_stories/stories/fffc5b49e126bc5489800e760aaf414d6680577c.story",
    "cnn_stories/stories/fffc660ed605dc82ba9f8de5b8e39e444002bf8f.story",
    "cnn_stories/stories/fffcaffda91f80b841efaefe04704e0357e4276c.story",
    "cnn_stories/stories/fffcd65676a501860ae312754e8cefc71f5ddab8.story",
    "cnn_stories/stories/fffce9eb5759655968713851647b4b29d19c7b29.story",
    "cnn_stories/stories/fffd170a9d15b1f9751e969e6f5b0ce5b9f7d027.story",
    "cnn_stories/stories/fffe0c4eb70bde9733b858adfd5b4eeeae631f28.story",
    "cnn_stories/stories/ffff11a2f44d731cd80c86819a89b7e227581415.story",
    "cnn_stories/stories/ffff2dc1cc4888253a4733f808959f0b4eab26a6.story",
    "cnn_stories/stories/ffff522cebe5ad9dcfb6dfc476b8f423f3f8dd34.story",
    ]

if __name__ == "__main__":
    for path in paths:
        with open(path) as cnn_story:
            story = p.tokenize(cnn_story.read())
            print(path, set(story))
