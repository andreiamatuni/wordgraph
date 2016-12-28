import json

import numpy as np
import networkx as nx

class WordGraph(object):
    def __init__(self, json_file=""):
        if json_file:
            self.graph = load_json_graph(json_file)

    def load_vector_model(self, vectors=None, vocab=None,
                          vectors_path="", vocab_path=""):
        """
        Load a word vector model.

        :param vectors: numpy ndarray with the word vectors
        :param vocab: dictionary of words to ndarray indices
        :param vectors_path: path to npy file with word vectors
        :param vocab_path: path to json formatted dictionary of
                            words to ndarray indices
        """
        if vectors and vocab:
            self.vectors = vectors
            self.vocab = vocab
        elif vectors_path and vocab_path:
            self.vectors = np.load(vectors)
            with open(vocab_path, "rU") as input:
                self.dict = json.load(input)



def load_json_graph(path):
    G = nx.Graph()
    with open(path, "rU") as input:
        json_data = json.load(input)
        for key, neighbors in json_data.items():
            G.add_node(key, word=key, size=1)
            if neighbors:
                for neighb in neighbors:
                    G.add_edge(key, neighb[0])
    return G

