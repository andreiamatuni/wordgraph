import unittest

import wordgraph as wg

class TestGenerateGraph(unittest.TestCase):
    def test_generate(self):
        words = ["apple", "bear", "clown",
                 "snow", "pool", "hummus",
                 "grape", "sword", "bug",
                 "python", "cup", "crown"]
        graph = wg.WordGraph(words=words)
        graph.load_vector_model(vectors_path='/Volumes/External_1/code/semspace/data/model/vectors_glove_42b_300.npy',
                                vocab_path='/Volumes/External_1/code/semspace/data/model/dict_glove_42b_300')
        graph.generate('cos', 0.5)
        graph.to_pickle("test2_out")

if __name__ == "__main__":
    unittest.main()