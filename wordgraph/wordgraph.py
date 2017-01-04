from . import simfunc
from . import vecmodel
from . import write
from . import load


try:
   import cPickle as pickle
except:
   import pickle

import pandas as pd

import powerlaw

temp_model = None

class WordGraph(object):
    sim_func_map = simfunc.sim_func_map
    generate_range_write = write.generate_range

    def __init__(self, json_file="", pickle_file="",
                 words=[]):
        if json_file:
            self.graph = load.load_json_graph(json_file)
            self.words = None
        elif pickle_file:
            self.load_pickle(pickle_file)
        elif words:
            self.words = words


    def generate(self, simil_func="", epsilon=0):
        """
        Generate the semantic graph given a similarity function and
        a similarity threshold.
        :param epsilon: similarity threshold
        :param sim_func: similarity function
        """
        if self.words is None and not self.model:
            raise ValueError("Initial lexicon not set. First set self.words")

        if simil_func not in self.sim_func_map:
            raise ValueError("Unknown similarity function: {}".format(simil_func))

        self.graph = self.sim_func_map[simil_func](self.model, epsilon, self.words)
        self.epsilon = epsilon

    def generate_all(self, simil_func="", epsilon=0):
        """
        Generate the semantic graph given a similarity function and
        a similarity threshold.
        :param threshold: similarity threshold
        :param sim_func:
        """

        if simil_func not in self.sim_func_map:
            raise ValueError("Unknown similarity function: {}".format(simil_func))

        words = list(self.model.vocab.keys())
        self.graph = self.sim_func_map[simil_func](self.model, epsilon, words)
        self.epsilon = epsilon

    def degree_distribution(self):
        """
        Get the distribution of vertex degrees

        :return: Series with all the vertices' degrees
        """
        degree_dist = self.graph.degree()
        df = pd.DataFrame(list(degree_dist.items()), columns = ['word', 'degree'])
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
        df = pd.DataFrame(data=list(degree_dist.items()), columns=['word', 'degree'])

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
        self.words = pd.read_csv(path)[column].str.replace("+", "-")\
            .str.replace(" ", "-").str.lower().dropna().unique()


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

    def unload_vector_model(self):
        """
        Unload the word vectors and dictionary. Useful for
        reducing RAM usage.
        :return:
        """
        self.model = None