import json

import numpy as np

from . import simfunc

class VectorModel(object):
    sim_func_map = simfunc.sim_func_map

    def __init__(self, vectors=None, vocab=None,
                 vectors_path="", vocab_path="",
                 rand=False, n=0, m=0, unit=False):
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
        elif rand:
            self.rand_unif_dist(n, m)

        self.unit = unit

    def rand_unif_dist(self, n, m):
        self.vectors = np.zeros((m,n))
        for i in range(m):
            x = np.random.normal(0, 1, n)
            W = np.sum(x**2)
            self.vectors[i] = x/W**(1/2)
        self.vocab = dict((i, i) for i in range(m))


def normalize_vectors(input, output):
    vectors = np.load(input)

    for x in range(vectors.shape[0]):
        vector = vectors[x, :]
        unit_vec = np.zeros(vector.shape)
        d1 = np.linalg.norm(vector)
        unit_vec = (vector.T / d1).T
        vectors[x, :] = unit_vec

    np.save(output, vectors)


def glove_to_numpy(input, vec_out, vocab_out):
    with open(input, "rU") as glove_file:
        vocab = {}
        n = 0
        dim = 0
        for line in glove_file:
            if n == 0:
                dim = len(line.split()[1:])
            n += 1
        glove_file.seek(0)
        print(dim)
        vectors = np.zeros((n, dim), dtype=np.float32)
        print(vectors.shape)
        for i, line in enumerate(glove_file):
            split_line = line.split()
            vocab[split_line[0]] = i
            vec = [float(x) for x in split_line[1:]]
            vectors[i,:] = np.array(vec)
    np.save(vec_out, vectors)
    with open(vocab_out, "wb") as out:
        json.dump(vocab, out)
    return vocab, vectors
