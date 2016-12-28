import unittest

import wordgraph

class TestLoadJson(unittest.TestCase):
    def test_load(self):
        graph = wordgraph.WordGraph(json_file='data/semgraph_all')
        print(graph.graph.nodes())
        print("hello")
        print()

if __name__ == "__main__":
    unittest.main()