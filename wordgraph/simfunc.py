import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from numpy import linalg as la
from timeit import default_timer as timer


def cosine_func(self, epsilon, words):
    G = nx.Graph()
    for idx, word in enumerate(words):
        if word in self.vocab:
            G.add_node(word)
        else:
            continue
        if idx == len(words) - 1:
            continue
        if self.unit:
            neighbors = cosine_neighbors_unit(
                self, epsilon, word, words[idx + 1:])
        else:
            w1_vector = self.vectors[self.vocab[word], :]

            for corp_word in words[idx + 1:]:
                if corp_word == word:
                    continue
                if corp_word in self.vocab:

                    d = 1 - np.dot(w1_vector, self.vectors[self.vocab[corp_word], :])\
                        / (la.norm(w1_vector) * la.norm(self.vectors[self.vocab[corp_word], :]))

                    if d <= epsilon:
                        G.add_edge(word, corp_word, cosine=d)

    return G


def cosine_neighbors(self, epsilon, word, corpus, G):
    if word not in self.vocab:
        return None
    else:
        w1_vector = self.vectors[self.vocab[word], :]

    for corp_word in corpus:
        if corp_word == word:
            continue
        if corp_word in self.vocab:

            d = 1 - np.dot(w1_vector, self.vectors[self.vocab[corp_word], :])\
                / (la.norm(w1_vector) * la.norm(self.vectors[self.vocab[corp_word], :]))

            if d <= epsilon:
                G.add_edge(word, corp_word, cosine=d)


def cosine_neighbors_unit(self, epsilon, word, corpus):
    w1_vector = self.vectors[self.vocab[word], :]
    neighbors = []

    for corp_word in corpus:
        if corp_word == word:
            continue
        if corp_word in self.vocab:

            d = 1 - np.dot(w1_vector, self.vectors[self.vocab[corp_word], :])

            if d <= epsilon:
                neighbors.append((corp_word, cosine))
    return neighbors


def euclid_func(self, epsilon, words):
    G = nx.Graph()
    for index, word in enumerate(words):
        if word in self.vocab:
            G.add_node(word)
        else:
            continue
        if index == len(words) - 1:
            continue
        neighbors = euclid_neighbors(self, epsilon, word, words[index + 1:])
        if neighbors:
            for neighb in neighbors:
                G.add_edge(word, neighb[0], {'cosine': neighb[1]})
    return G


def euclid_neighbors(self, epsilon, word, corpus):
    w1_vector = self.vectors[self.vocab[word], :]
    neighbors = []

    for corp_word in corpus:
        if corp_word == word:
            continue
        if corp_word in self.vocab:
            w2_vector = self.vectors[self.vocab[corp_word], :]

            w1_vec_norm = np.zeros(w1_vector.shape)

            d1 = (np.sum(w1_vector ** 2, ) ** (0.5))
            w1_vec_norm = (w1_vector.T / d1).T

            w2_vec_norm = np.zeros(w2_vector.shape)
            d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
            w2_vec_norm = (w2_vector.T / d2).T

            euclid = np.linalg.norm(w1_vec_norm.T - w2_vec_norm.T)

            if 1 - euclid <= epsilon:
                neighbors.append((corp_word, euclid))
    return neighbors


def dotprod_func(self, epsilon, words):
    G = nx.Graph()
    for index, word in enumerate(words):
        if word in self.vocab:
            G.add_node(word)
        else:
            continue
        if index == len(words) - 1:
            continue

        neighbors = dotprod_neighbors(self, epsilon, word, words[index + 1:])
        if neighbors:
            for neighb in neighbors:
                G.add_edge(word, neighb[0], {'dotprod': neighb[1]})
    return G


def dotprod_neighbors(self, epsilon, word, corpus):
    if word not in self.vocab:
        return None
    else:
        w1_vector = self.vectors[self.vocab[word], :]
        neighbors = []

    for corp_word in corpus:
        if corp_word == word:
            continue
        if corp_word in self.vocab:
            w2_vector = self.vectors[self.vocab[corp_word], :]

            dotprod = np.dot(w1_vector, w2_vector)

            if dotprod >= epsilon:
                neighbors.append((corp_word, dotprod))
    return neighbors


sim_func_map = {
    'cos': cosine_func,
    'euc': euclid_func,
    'dot': dotprod_func
}
