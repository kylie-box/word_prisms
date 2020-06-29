import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import glob

from dictionary import Dictionary
from embedding_models import Embeddings
import codecs

"""
from pretrained embedding to the word prism form
"""


def load_embeddings(emb_dir):
    emb_file = \
            [f for f in os.listdir(emb_dir) if f.endswith('vec') or f.endswith('txt')][0]
    emb_file = os.path.join(emb_dir, emb_file)
    print("Loading {}".format(emb_file))
    f = open(emb_file, 'r')
    num_lines = sum(1 for _ in f)

    dictionary = []
    embeddings = []

    if emb_file.endswith('vec'):
        # remove the first line for fasttext
        next(f)

    with open(emb_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            splitLine = line.split()
            word = splitLine[:-300]
            # if len(word) > 1:
            #     import pdb; pdb.set_trace()
            dictionary.append(" ".join(word))
            embedding = np.array([float(val) for val in splitLine[-300:]])
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    print("last loaded word ", word)
    assert len(dictionary) == len(embeddings)
    print("Done.", len(dictionary), " words loaded!")

    return dictionary, embeddings


def save_embeddings(emb, system_name=None, dictionary=None, path=None):
    if path is None:
        path = os.path.dirname(dictionary.dict_path)
    print("path is: ", path)
    saved_shape = Embeddings.save_embedding_from_V(path, emb)
    with codecs.open(os.path.join(path, "dictionary"), 'w', 'utf8') as f:
        f.write('\n'.join(dictionary.tokens))
    print("saved embedding shape: ", saved_shape)
    print("saved embeddings for system: ", system_name)


def save_dictionary(tokens, save_dir):
    dictionary = Dictionary(tokens=tokens)
    print("the length of the dictionary loaded: ", len(dictionary))
    dictionary.set_dictionary_path(save_dir)
    dictionary.save()
    return dictionary


def read_given_dict(path):
    with open(path, 'r') as f:
        d = f.readlines()
    d = [tok.strip('\n') for tok in d]
    return d


def main(args):

    tokens, embeddings = load_embeddings(args.emb_path)
    dictionary = save_dictionary(tokens, args.emb_path)
    system_name = os.path.split(args.emb_path)[1]
    save_embeddings(embeddings, system_name=system_name, dictionary=dictionary)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--emb_path', type=str,
                        help='full path to the directory named as the system name that contains the .txt or .vec file')
    args = parser.parse_args()
    main(args)
