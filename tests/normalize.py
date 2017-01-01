import unittest
import os

import wordgraph as wg


glove_vector_path = '/Volumes/External_1/code/semspace/data/model/vectors_glove_42b_300.npy'
glove_vocab_path = '/Volumes/External_1/code/semspace/data/model/dict_glove_42b_300'

class TestNormalize(unittest.TestCase):
    def test_normalize(self):
        output_base = os.path.dirname(glove_vector_path)
        output = os.path.join(output_base, 'glove_unit_42b_300.npy')
        wg.normalize_vectors(glove_vector_path, output)



if __name__ == "__main__":
    unittest.main()