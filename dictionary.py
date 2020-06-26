from copy import deepcopy
import subprocess
import codecs
import os
import torch


"""
Most of the wp are copied from the hilbert codebase.
Added modification to accommodate the word prism requirements
"""


class Dictionary(object):
    """
    token to myid map
    word prism id to myid map
    """
    def __init__(self,
                 tokens=None,
                 dictionary_path=None):
        self.tokens = []
        self.token_ids = {}
        # Don't know wpid until the word prism is initialized
        self.wpid_myid = None
        self.dict_path = dictionary_path
        if dictionary_path is not None:
            self.load(dictionary_path)

        if tokens is not None:
            for token in tokens:
                self.add_token(token)

        self.myids = list(self.token_ids.values())
        assert len(self.myids) == len(self.tokens)
        self.unk_id = len(self)
        self.padding_id = len(self) + 1


    def __copy__(self):
        return deepcopy(self)


    def __contains__(self, key):
        return key in self.token_ids


    def __deepcopy__(self, memo):
        result = Dictionary(self.tokens)
        memo[id(self)] = result
        return result

    def __len__(self):
        return len(self.tokens)

    def set_dictionary_path(self, path):
        self.dict_path = os.path.join(path, 'dictionary')


    def load_from_tokens(self):
        self.token_ids = {
            token: idx
            for idx, token in enumerate(self.tokens)
        }

    def truncation(self, thres=None):
        if self.wpid_myid is not None:
            raise ValueError('wpid linkage is not empty! Don\'t truncate')
        if thres >= len(self):
            return self
        new_tokens = self.tokens[:thres]
        return Dictionary(tokens=new_tokens)

    def load(self, path):
        path = os.path.join(path, "dictionary")
        with open(path) as f:
            self.tokens = f.read().split('\n')
            self.load_from_tokens()

    def add_token(self, token):
        if token not in self.token_ids:
            idx = len(self.tokens)
            self.token_ids[token] = idx
            self.tokens.append(token)
        else:
            idx = len(self.tokens)
            while token in self.token_ids:
                # a hack to deal with the repeated vocab in glove
                token = token + " "
            self.token_ids[token] = idx
            self.tokens.append(token)


    def save(self):
        if not os.path.exists(self.dict_path):
            os.makedirs(os.path.split(self.dict_path)[0])
        with codecs.open(self.dict_path, 'w', 'utf8') as f:
            f.write('\n'.join(self.tokens))

    @staticmethod
    def save_from_dictionary(path, tokens):
        """
        The same as ``save()'', but without creating the Dictionary object

        :param path: path to save
        :param tokens: dictionary
        :return:
        """
        with codecs.open(path, 'w', 'utf8') as f:
            f.write('\n'.join(tokens))

    def get_token_from_myid(self, myid):
        if myid == self.unk_id:
            return '<unk>'
        elif myid == self.padding_id:
            return '<pad>'
        return self.tokens[myid]

    def get_token_from_wpid(self, wpid):
        return self.get_token_from_myid(self.get_myid_from_wpid(wpid))

    def get_id(self, token):
        return self.get_myid_from_token(token)

    def get_myid_from_token(self, token):
        # for prism then means get wpid_from_token
        if token in self.token_ids:
            return self.token_ids[token]
        else:
            # for wp oov it will return the self.unk_id
            return self.unk_id

    def get_myid_from_wpid(self, wpid):
        return self.wpid_myid[wpid]

    def check_myid(self, myid):
        if myid is None:
            return None
        return myid in self.myids

    def link_word_prism_id(self, wp_d, device='cpu'):
        """
        generate a dictionary of mapping from facet dictionary id to word prism
        dictionary id
        the wpid_myid is a tensor of indices mapping

        :param wp_d: Dictionary of word prism


        """
        id_map = torch.zeros((len(wp_d) + 2,), dtype=torch.long)
        for token in wp_d.tokens:
            facet_id = self.get_myid_from_token(token)
            wp_id = wp_d.get_myid_from_token(token)
            # wp's dictionary should have one to one mapping in myid_wpid
            if facet_id is None:
                # unk token in this facet
                facet_id = len(self)
            id_map[wp_id] = facet_id
        # len(wp_d) is unk; len(wp_d) + 1 is padding of word prism
        # len(f_d) is unk for this facet; len(f_d) + 1 is the padding of facet
        # wp_unk --> f_unk; wp_pad --> f_pad; wp_None --> f_unk
        id_map[len(wp_d)] = len(self.tokens) # wp_unk
        id_map[len(wp_d) + 1] = len(self.tokens) + 1 # wp_pad
        self.wpid_myid = id_map.to(device)


    @staticmethod
    def check_vocab(path):
        """
        Determine the vocabulary of a dictionary on disk without creating a
        Dictionary instance instance.
        """
        # We can tell the vocabulary from how long the ditionary
        result = subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE)
        num_lines = int(result.stdout.split()[0]) + 1
        return num_lines

    def get_indices(self, wp_indices):
        """
        slice of my ids
        :param wp_indices: a indexing tensor to retrive the my ids
        :return: my_ids as a tensor
        """
        return self.wpid_myid[wp_indices]



