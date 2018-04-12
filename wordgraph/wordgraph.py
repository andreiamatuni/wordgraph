from . import simfunc
from . import vecmodel
from . import write
from . import load


try:
    import cPickle as pickle
except:
    import pickle

import pandas as pd
import networkx as nx
import numpy as np
import powerlaw
import scipy

temp_model = None


class WordGraph(object):
    sim_func_map = simfunc.sim_func_map
    generate_range_write = write.generate_range

    def __init__(self, json_file="", pickle_file="",
                 words=[], graph=None):
        if json_file:
            self.graph = load.load_json_graph(json_file)
            self._words = None
        elif pickle_file:
            self.load_pickle(pickle_file)
        elif words:
            self._words = words
        elif graph:
            self._words = list(graph.nodes)
            self.graph = graph

    def words(self):
        return list(self.graph.nodes)

    def generate(self, simil_func="", epsilon=0):
        """
        Generate the semantic graph given a similarity function and
        a similarity threshold.
        :param epsilon: similarity threshold
        :param sim_func: similarity function
        """
        if self._words is None and not self.model:
            raise ValueError("Initial lexicon not set. First set self._words")

        if simil_func not in self.sim_func_map:
            raise ValueError(
                "Unknown similarity function: {}".format(simil_func))

        self.graph = self.sim_func_map[simil_func](
            self.model, epsilon, self._words)
        self._words = list(self.graph.nodes)
        self.epsilon = epsilon

    def generate_all(self, simil_func="", epsilon=0):
        """
        Generate the semantic graph given a similarity function and
        a similarity threshold.
        :param threshold: similarity threshold
        :param sim_func:
        """

        if simil_func not in self.sim_func_map:
            raise ValueError(
                "Unknown similarity function: {}".format(simil_func))

        words = list(self.model.vocab.keys())
        self.graph = self.sim_func_map[simil_func](self.model, epsilon, words)
        self.epsilon = epsilon

    def degree_distribution(self):
        """
        Get the distribution of vertex degrees

        :return: Series with all the vertices' degrees
        """
        degree_dist = self.graph.degree()
        df = pd.DataFrame([(x, y) for x, y in dict(degree_dist).items()],
                          columns=['word', 'degree'])
        return df

    def simil_distribution(self):
        edges = self.graph.edges()
        return [self.graph.edge[x][y]['cosine'] for x, y in edges]

    def top_degree(self,  n=0, all=False):
        """
        Get the top N nodes in the graph by degree
        :param n: number of top degree nodes to pull
        :param all: return all of them
        :return: DataFrame of word and degree, sorted by degree
        """
        degree_dist = self.graph.degree()
        df = pd.DataFrame([(x, y) for x, y in dict(degree_dist).items()],
                          columns=['word', 'degree'])

        if all:
            return df
        else:
            return df.sort_values('degree', ascending=False)[:n]

    def fit_power_law(self, discrete=False, xmin=None, xmax=None,
                      fit_method='Likelihood', estimate_discrete=True,
                      discrete_approximation='round', sigma_threshold=None,
                      parameter_range=None, fit_optimizer=None, xmin_distance='D',
                      **kwargs):
        args = locals()
        del args['self']
        degree_dist = self.degree_distribution()
        result = powerlaw.Fit(degree_dist['degree'] + 1, **args)
        return result

    def to_pickle(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """
        Dump the WordGraph to a pickle file

        :param path: path to output
        :param protocol: pickle protocol
        :return:
        """
        temp_model = self.model
        self.model = None
        with open(path, "wb") as out:
            pickle.dump(self, out, protocol)
        self.model = temp_model
        temp_model = None

    def load_pickle(self, path):
        """
        Load a pickled WordGraph and update the current instance
        with it.

        :param path: path to pickled WordGraph object
        :return:
        """
        with open(path, 'rb') as input:
            temp_dict = pickle.load(input)
            self.__dict__.update(temp_dict.__dict__)

    def load_csv_words(self, path="", column=""):
        """
        Load your words from a csv file, given a column to
        select from the file. This function assumes the
        csv file is properly formatted.

        :param path: path to csv file
        :param column: the column with the words you want
        :return:
        """
        self._words = pd.read_csv(path, low_memory=False)[column]\
            .str.strip().str.replace("+", "-")\
            .str.replace(" ", "-").str.lower().dropna().unique()

    def load_words(self, words, column=''):
        if column:
            self._words = words[column].str.replace("+", "-")\
                .str.replace(" ", "-").str.lower().dropna().unique()
        else:
            self._words = words

    def load_vector_model(self, vectors=None, vocab=None,
                          vectors_path="", vocab_path="",
                          rand=False, n=0, m=0, unit=False):
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
                                          vocab_path,
                                          rand, n, m, unit)

    def average_shortest_path(self):
        """
        Get the average shortest path length of the largest
        connected subgraph.
        :return: avg. shortest path length
        """
        if self.graph is not None:
            largest = sorted(nx.connected_components(self.graph),
                             key=len, reverse=True)[0]
            subgraph = self.graph.subgraph(largest)
            return nx.average_shortest_path_length(subgraph)

    def average_clustering(self):
        """
        Get the clustering coefficient for the largest connected
        subgraph
        :return:
        """
        if self.graph is not None:
            largest = sorted(nx.connected_components(self.graph),
                             key=len, reverse=True)[0]
            subgraph = self.graph.subgraph(largest)
            return nx.average_clustering(subgraph)

    def conductance(self, words):
        return nx.algorithms.conductance(self.graph, words)

    def adj_matrix(self, subgraph=None):
        """
        Return the adjacency matrix for the graph.

        :param subgraph: only return adjacency matrix for 
                         a subgraph specified by these specific nodes
        """
        if subgraph is not None:
            g = self.graph.subgraph(subgraph)
            return nx.linalg.adj_matrix(g), list(g.nodes)
        return nx.linalg.adj_matrix(self.graph), list(self.graph.nodes)

    def trans_matrix(self, subgraph=None):
        """
        Return the transition probability matrix associated to
        this graph, assuming a uniform distribution for all out-degrees
        for all vertices
        """
        A, words = self.adj_matrix(subgraph=subgraph)
        return adj_to_trans(A), words

    def eigen(self, subgraph=None):
        """
        Return the eigenvalues and eigenvectors of the graph

        :param subgraph: only look at specified subgraph (list of nodes)
        """
        P, words = self.trans_matrix(subgraph=subgraph)
        w, v = scipy.linalg.eig(P, left=True, right=False)
        return w, v, words

    def spectral_gap(self, subgraph=None):
        w, v = self.eigen(subgraph=subgraph)
        w = np.apply_along_axis(lambda x: abs(x), 0, w)
        idx = w.argsort()[::-1]
        w = w[idx]
        return w[0] - w[1]

    def subgraph(self, nodes):
        g = self.graph.subgraph(nodes)
        G = WordGraph()
        G.graph = g
        G._words = list(nodes)
        return G

    def max_connected_component(self):
        nodes = max(nx.connected_components(self.graph), key=len)
        return self.subgraph(nodes)

    def max_connected_nodes(self):
        return max(nx.connected_components(self.graph), key=len)

    def pi(self, x, p=None):
        """
        Return probability of being at word x under the stationary 
        distribution for Markov chain with uniform transition probability 
        matrix. The ordering in pi(x) is such that indexes correspond to 
        their ordering in self.graph.nodes

        :param x: word for which you want pi(x)
        :param p: the stationary distribution vector (computed if not provided)
        """
        if p is None:
            p = self.stationary_distribution()

        i = list(self.graph.nodes).index(x)
        return p[i]

    def stationary_distribution(self):
        vals, vecs, words = self.eigen(subgraph=self.max_connected_nodes())
        vals = np.apply_along_axis(lambda x: abs(x), 0, vals)
        idx = vals.argsort()[::-1]
        vecs = vecs[idx]
        renormed_p = vecs[0] / sum(vecs[0])
        return renormed_p, words

    def unload_vector_model(self):
        """
        Unload the word vectors and dictionary. Useful for
        reducing RAM usage.
        :return:
        """
        self.model = None


def adj_to_trans(A):
    """
    Convert an adjacency matrix into a transition matrix
    with uniform transition probabilities.
    """
    return np.apply_along_axis(lambda i: i * (1 / sum(i)), 1, A.A)
