from . import simfunc
from . import vecmodel

import json

import numpy as np
import networkx as nx

class WordGraph(object):
    sim_func_map = simfunc.sim_func_map

    def __init__(self, json_file="", words=[]):
        if json_file:
            self.graph = load_json_graph(json_file)
            self.words = None
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

        self.sim_func_map[simil_func](self.model, epsilon, self.words)


    def to_pickle(self, path):
        nx.write_gpickle(self.graph, path)

    def load_pickle(self, path):
        self.graph = nx.read_gpickle(path)

    def load_vector_model(self, vectors=None, vocab=None,
                          vectors_path="", vocab_path=""):

        self.model = vecmodel.VectorModel(vectors, vocab,
                                          vectors_path,
                                          vocab_path)



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

