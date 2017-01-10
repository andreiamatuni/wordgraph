import unittest
import wordgraph as wg


glove_vector_path = '/Volumes/External_1/code/semspace/data/model/glove_unit_42b_300.npy'
glove_vocab_path = '/Volumes/External_1/code/semspace/data/model/dict_glove_42b_300'


class TestRangeGenerate(unittest.TestCase):

    def test_generate(self):
        graph = wg.WordGraph()
        graph.load_csv_words('data/some_words.csv', column='word')

        graph.load_vector_model(vectors_path=glove_vector_path,
                                vocab_path=glove_vocab_path, unit=True)

        graph.generate_range_write(output_dir="output/regular_words", simil_func='cos',
                                   start=0.0, end=0.5, step=0.01)

        print("hello")
