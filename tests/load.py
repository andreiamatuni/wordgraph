import unittest

import wordgraph as wg

class TestLoadJson(unittest.TestCase):
    def test_load(self):
        graph = wg.WordGraph(json_file='data/semgraph_all')
        # graph = wg.WordGraph()
        # graph.load_pickle('test2_out')
        top = graph.top_degree(n=20)

        print(graph.nodes)




if __name__ == "__main__":
    unittest.main()