import unittest
import wordgraph as wg


glove_vector_path = '/Users/andrei/code/work/bergelsonlab/semantic_nets/model/glove_unit_42b_300.npy'
glove_vocab_path = '/Users/andrei/code/work/bergelsonlab/semantic_nets/model/dict_glove_42b_300'


class TestRangeGenerate(unittest.TestCase):

    def test_generate(self):
        G = wg.WordGraph()
        G.load_csv_words('~/BLAB_DATA/all_basiclevel/all_basiclevel.csv', column='basic_level')

        G.load_vector_model(vectors_path=glove_vector_path,
                                vocab_path=glove_vocab_path)

        G.generate_range_write(output_dir="output/regular_words", simil_func='cos',
                                   start=0.01, end=0.5, step=0.01)

        print("hello")
