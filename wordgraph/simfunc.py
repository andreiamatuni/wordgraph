import networkx as nx
import numpy as np

def cosine_func(self, epsilon, words):
    G = nx.Graph()
    for index, word in enumerate(words):
        G.add_node(word)
        if index == len(words) - 1:
            continue
        neighbors = cosine_neighbors(self, epsilon, word, words[index + 1:])
        if neighbors:
            for neighb in neighbors:
                G.add_edge(word, neighb[0])
    return G

def cosine_neighbors(self, epsilon, word, corpus):
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

            w1_vec_norm = np.zeros(w1_vector.shape)
            d1 = (np.sum(w1_vector ** 2, ) ** (0.5))
            w1_vec_norm = (w1_vector.T / d1).T

            w2_vec_norm = np.zeros(w2_vector.shape)
            d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
            w2_vec_norm = (w2_vector.T / d2).T

            cosine = np.dot(w1_vec_norm.T, w2_vec_norm.T)

            if 1 - cosine <= epsilon:
                neighbors.append((corp_word, cosine))
    return neighbors

sim_func_map = {
    'cos': cosine_func
}
