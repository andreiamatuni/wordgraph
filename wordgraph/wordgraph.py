from . import simfunc
from . import vecmodel
from .load import *

try:
   import cPickle as pickle
except:
   import pickle

import pandas as pd

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
        if self.words is None:
            raise ValueError("Initial lexicon not set. First set self.words")

        if simil_func not in self.sim_func_map:
            raise ValueError("Unknown similarity function: {}".format(simil_func))

        self.graph = self.sim_func_map[simil_func](self.model, epsilon, self.words)
        self.epsilon = epsilon


    def degree_ditribution(self):
        """
        Get the distribution of vertex degrees

        :return: Series with all the vertices' degrees
        """
        degree_dist = self.graph.degree()
        dist = pd.Series(degree_dist.values())
        return dist


    def top_degree(self,  n=0, all=False):
        """
        Get the top N nodes in the graph by degree
        :param n: number of top degree nodes to pull
        :param all: return all of them
        :return: DataFrame of word and degree, sorted by degree
        """
        degree_dist = self.graph.degree()
        df = pd.DataFrame(data=degree_dist.items(), columns=['word', 'degree'])

        if all:
            return df
        else:
            return df.sort_values('degree', ascending=False)[:n]


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
        self.words = pd.read_csv(path)[column].str.lower().dropna().unique()


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
