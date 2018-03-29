import wordgraph as wg
from timeit import default_timer as timer
import profile


def main():
    glove_vector_path = '/Users/andrei/code/work/bergelsonlab/semantic_nets/model/glove_unit_42b_300.npy'
    glove_vocab_path = '/Users/andrei/code/work/bergelsonlab/semantic_nets/model/dict_glove_42b_300'

    G = wg.WordGraph()
    G.load_csv_words(
        'data/all_basiclevel_super_reduced.csv',
        column='basic_level')

    G.load_vector_model(vectors_path=glove_vector_path,
                        vocab_path=glove_vocab_path)

    start = timer()

    G.generate_range_write(output_dir="output/regular_words", simil_func='cos',
                           start=0.6, end=0.61, step=0.01)
    end = timer()

    print("\ntime elapsed : {} seconds".format(round(end - start, 2)))


if __name__ == "__main__":
    # profile.run('main()')
    main()