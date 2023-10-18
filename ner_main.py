import sys
import utils.functions as fs
import fasttext.util as util
import fasttext
import numpy as np


def load_vectors(fname):
    data = {}
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, dim = map(int, fin.readline().split())
        for line in fin:
            tokens = line.rstrip().split(' ')
            word, vector = tokens[0], tokens[1:]
            if dim != len(vector):
                continue
            data[word] = map(float, vector)
        return data, n, dim

def load_vectors_alt(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        n, embedd_dim = map(int, file.readline().split())
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


if __name__ == '__main__':
    print('loading vectors...')
    data, dim = load_vectors_alt('fasttext/cc.he.300.vec')
    print('loaded vectors')
    # print('loading model...')
    # ft = fasttext.load_model('fasttext/cc.he.300.bin')
    # print('loaded model')
    print('vector version')
    print(dim)
    print(data.get('ליברציה'))
    # print('model version')
    # print(ft.get_dimension())
    # print(ft.get_word_vector('ליברציה'))
    # print('nearest neighbours to שלום')
    # print(ft.get_nearest_neighbors('שלום'))