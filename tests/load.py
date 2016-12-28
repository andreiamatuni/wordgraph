import unittest

import wordgraph

class TestLoadJson(unittest.TestCase):
    def test_load(self):
        graph = wordgraph.WordGraph(json_file='data/semgraph_all')

class TestGenerateGraph(unittest.TestCase):
    def test_generate(self):
        words = ["apple", "bear", "clown",
                 "snow", "pool", "hummus",
                 "grape", "sword", "bug",
                 "python", "cup", "crown"]
        graph = wordgraph.WordGraph(words=words)
        graph.generate('cos', 0.5)



if __name__ == "__main__":
    unittest.main()