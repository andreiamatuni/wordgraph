import networkx as nx
import numpy as np

def cosine_func(model, epsilon, words):
    G = nx.Graph()
    for index, word in enumerate(words):
        G.add_node(word)
        if index == len(words) - 1:
            continue
        result = cosine_neighbors(model, epsilon)
        neighbors = model.neighbor_density_cos(epsilon, word, words[index + 1:])
        if neighbors:
            for neighb in neighbors:
                G.add_edge(word, neighb[0])

def cosine_neighbors(model, epsilon, word, corpus):
    if word not in model.vocab:
        return None
    else:
        w1_vector = model.vectors[model.dict[word], :]
        distances = []

    for corp_word in corpus:
        if corp_word == word:
            continue
        if corp_word in model.dict:
            if corp_word == word:
                continue
            w2_vector = model.vectors[model.vocab[corp_word], :]

            w1_vec_norm = np.zeros(w1_vector.shape)
            d1 = (np.sum(w1_vector ** 2, ) ** (0.5))
            w1_vec_norm = (w1_vector.T / d1).T

            w2_vec_norm = np.zeros(w2_vector.shape)
            d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
            w2_vec_norm = (w2_vector.T / d2).T

            dist = np.dot(w1_vec_norm.T, w2_vec_norm.T)

            if 1 - dist <= epsilon:
                distances.append((corp_word, dist))
    return distances

sim_func_map = {
    'cos': cosine_func
}
