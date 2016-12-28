import pickle
import json

import networkx as nx

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
