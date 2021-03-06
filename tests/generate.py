import unittest

import wordgraph as wg

glove_vector_path = '/Volumes/External_1/code/semspace/data/model/glove_unit_42b_300.npy'
glove_vocab_path = '/Volumes/External_1/code/semspace/data/model/dict_glove_42b_300'

class TestGenerateGraph(unittest.TestCase):
    def test_generate(self):
        graph = wg.WordGraph()
        graph.load_csv_words('data/01_13.csv', column='basic_level')
        graph.load_vector_model(vectors_path=glove_vector_path,
                                vocab_path=glove_vocab_path)
        graph.generate('cos', 0.5)
        graph.to_pickle("output/test_out")
        print()

if __name__ == "__main__":
    unittest.main()