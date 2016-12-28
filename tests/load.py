import unittest

import wordgraph as wg

class TestLoadPickle(unittest.TestCase):
    def test_load(self):
        graph = wg.WordGraph()
        graph.load_pickle('output/test_out')
        print(graph.graph.nodes())


if __name__ == "__main__":
    unittest.main()