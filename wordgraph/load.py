try:
    import cPickle as pickle
except:
    import pickle

import json

import networkx as nx

from . import wordgraph as wg


def load_pickle(path):
    """
    Load a WordGraph object from a pickle file.

    :param path: path to pickled WordGraph object
    :return:
    """
    with open(path, 'rb') as input:
        # G = wg.WordGraph()

        return pickle.load(input)


def json_graph(path):
    G = nx.Graph()
    with open(path, "rU") as input:
        json_data = json.load(input)
        for key, neighbors in json_data.items():
            G.add_node(key, word=key, size=1)
            if neighbors:
                for neighb in neighbors:
                    G.add_edge(key, neighb[0])
    return G
