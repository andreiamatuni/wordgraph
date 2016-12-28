from . import simfunc
from . import vecmodel

import json
import pickle

import numpy as np
import networkx as nx

class WordGraph(object):
    sim_func_map = simfunc.sim_func_map

    def __init__(self, json_file="", pickle_file="",
                 words=[]):
        if json_file:
            self.graph = load_json_graph(json_file)
            self.words = None
        elif pickle_file:
            self.load_pickle(pickle_file)
        if words:
            self.words = words

    def generate(self, simil_func="", epsilon=0):
        """
        Generate the semantic graph given a similarity function and
        a similarity threshold.
        :param threshold:
        :param sim_func:
        """
        if not self.words:
            raise ValueError("Initial lexicon not set. First set self.words")

        if simil_func not in self.sim_func_map:
            raise ValueError("Unknown similarity function: {}".format(simil_func))

        self.graph = self.sim_func_map[simil_func](self.model, epsilon, self.words)
        self.epsilon = epsilon


    def to_pickle(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """
        Dump the WordGraph to a pickle file

        :param path: path to output
        :param protocol: pickle protocol
        :return:
        """

        self.model = None
        with open(path, "wb") as out:
            pickle.dump(self, out, protocol)

    def load_pickle(self, path):
        """
        Load a pickled WordGraph and update the current instance
        with it.

        :param path: path to pickled WordGraph object
        :return:
        """
        with open(path, 'r') as input:
            temp_dict = pickle.load(input)
            self.__dict__.update(temp_dict.__dict__)

    def load_vector_model(self, vectors=None, vocab=None,
                          vectors_path="", vocab_path=""):
        """
        Load a word vector model. It will be set to self.model

        :param vectors: numpy ndarray with the word vectors
        :param vocab: dictionary of words to ndarray indices
        :param vectors_path: path to npy file with word vectors
        :param vocab_path: path to json formatted dictionary of
                            words to ndarray indices
        """

        self.model = vecmodel.VectorModel(vectors, vocab,
                                          vectors_path,
                                          vocab_path)


def load_pickle(path):
    """
    Load a WordGraph object from a pickle file.

    :param path: path to pickled WordGraph object
    :return:
    """
    with open(path, 'r') as input:
        return pickle.load(input)

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

