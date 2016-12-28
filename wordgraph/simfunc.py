import networkx as nx

def cosine_grapher(model, thresh, words):
    G = nx.Graph()
    for index, word in enumerate(words):
        G.add_node(word)
        if index == len(words) - 1:
            continue
        density_map[word] = model.neighbor_density_cos(threshold, word, words[index + 1:])

sim_func_map = {
    'cos': cosine_grapher
}
