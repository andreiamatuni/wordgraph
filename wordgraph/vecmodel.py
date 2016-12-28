import json

import numpy as np

from . import simfunc

class VectorModel(object):
    sim_func_map = simfunc.sim_func_map

    def __init__(self, vectors=None, vocab=None,
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
            self.vectors = np.load(vectors_path)
            with open(vocab_path, "rU") as input:
                self.vocab = json.load(input)