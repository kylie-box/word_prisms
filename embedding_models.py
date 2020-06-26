from collections import OrderedDict

import numpy as np
import os
import torch
import torch.nn.functional as F
import utils
from dictionary import Dictionary
from sklearn.decomposition import PCA
from torch import nn


class Embeddings:

    def __init__(self,
                 input_V,
                 bias=None,
                 device=None,
                 dictionary=None,
                 system_name=None,
                 truncation=None,
                 normalize=False,
                 ):
        self.device = utils.get_device(device)
        self.dtype = utils.get_dtype()
        self.system_name = system_name
        if truncation is not None:
            _V = torch.tensor(input_V[:truncation], dtype=self.dtype,
                              device=self.device)
            self.dictionary = dictionary.truncation(thres=truncation)
        else:
            _V = torch.tensor(input_V, dtype=self.dtype, device=self.device)
            self.dictionary = dictionary
        _bias = (None if bias is None else torch.tensor(
            bias, dtype=self.dtype, device=self.device))
        # self.bias = (None if _bias is None
        #              else torch.cat(
        #     (_bias, torch.tensor([0.], device=self.device)), dim=0))
        self.bias = _bias
        self.dim = _V.shape[1]
        self.unk = torch.mean(_V, dim=0)
        self.pad = torch.zeros(self.dim, device=self.device)
        # _V = torch.cat((_V, self.unk.unsqueeze(0)), dim=0)

        # check if normalized

        self.normalized = False
        if normalize:
            _V = self.normalize(_V)

        # final check
        self.validate(_V)

        # padding index of the current word system gets the index of the length
        # of the dictionary + 1
        self.V = nn.Embedding(len(self.dictionary) + 2, self.dim,
                              padding_idx=len(self.dictionary) + 1)
        # all vocab idx; unk_idx; padding_idx
        self.V.weight = nn.Parameter(
            torch.cat((_V, self.unk.unsqueeze(0), self.pad.unsqueeze(0)),
                      dim=0),
            requires_grad=False)

        del _V  # release some space

    def validate(self, _V):
        assert _V.shape[1] == self.dim
        if self.bias is not None and _V.shape[0] != self.bias.shape[0]:
            raise ValueError(
                'The number of vector embeddings and vector biases do not '
                'match.  Got {} vectors and {} biases'.format(
                    _V.shape[0], self.bias.shape[0]
                )
            )

        assert self.dim == self.unk.shape[0]
        # we dont't store unk embedding in _V, add unk ad hoc
        if _V.shape[0] != len(self.dictionary):
            raise ValueError(
                "Number of embeddings {} does not match the number of"
                " words {} in the dictionary.".format(_V.shape[0],
                                                      len(self.dictionary)))

    def check_normalized(self, _V):
        """
        see if normalized in vector

        """
        ones = torch.ones(_V.shape[0] - 1, device=self.device)
        V_normed = torch.allclose(utils.norm(_V[:-1], axis=1), ones)
        self.normalized = V_normed

    def normalize(self, _V):
        """
        Normalize the vectors if they aren't already normed.
        """

        return self._normalize(_V)

    def _normalize(self, _V):
        """
        Normalize the vectors.
        """
        _V[:-1] = utils.normalize(_V[:-1], axis=1)
        self.normalized = True
        return _V

    @staticmethod
    def load_embedding(path, truncation, normalize=False, device=None, random_cnt=0):
        """
        Static method for loading embeddings stored at ``path``.

        """

        V, bias, dictionary = None, None, None

        if random_cnt > 0:
            dictionary = Dictionary(dictionary_path=path)
            system_name = "random_{}".format(random_cnt)
            V = np.random.uniform(low=-0.2, high=0.2, size=(500000, 300))
        else:
            if os.path.exists(path):
                dictionary = Dictionary(dictionary_path=path)
            system_name = os.path.split(path)[-1]
            V = np.load(os.path.join(path, 'V.npy'))
            if normalize:
                V = F.normalize(torch.tensor(V, device=device), p=2, dim=1)

            if os.path.exists(os.path.join(path, 'bias.npy')):
                bias = np.load(os.path.join(path, 'bias.npy'))

        return Embeddings(
            V, bias=bias, device=device, dictionary=dictionary,
            system_name=system_name, truncation=truncation, normalize=normalize
        )

    @staticmethod
    def save_embedding_from_V(path, input_V, input_bias=None):
        """
        The same as save embedding, without creating the embedddings object

        :param path: path to save
        :param input_V: embedding matrix
        :param input_bias: bias vector in numpy array
        :return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        V = np.array(input_V, dtype=np.float32)
        if input_bias is not None:
            bias = np.array(input_bias, dtype=np.float32)
        np.save(os.path.join(path, 'V.npy'), V)
        if input_bias is not None:
            np.save(os.path.join(path, 'bias.npy'), bias)
        return V.shape

    def save_embedding(self, path):
        """
        save embedding
        :param path:
        :return: None
        """

        if not os.path.exists(path):
            os.makedirs(path)
        # remove the last row of unk
        np.save(os.path.join(path, 'V.npy'), self.V.weight[:-1].cpu().numpy())
        if self.bias is not None:
            np.save(os.path.join(path, 'bias.npy'), self.bias.cpu().numpy())
        if self.dictionary is not None:
            self.dictionary.save()

    def get_vec_from_myid(self, myid):
        if self.dictionary.check_myid(myid):
            return self.V.weight[myid]
        else:
            return self.unk

    def projection(self, out_dim):
        emb_size = self.embeddings.dim
        assert emb_size >= out_dim
        pca = PCA(n_components=out_dim)

        self.embeddings.V = torch.tensor(
            pca.fit_transform(self.embeddings.V.numpy()),
            dtype=self.embeddings.dtype,
            device=self.embeddings.device)
        self.embeddings.dim = out_dim
        self.dim = out_dim


class EmbWrapper:
    def __init__(self,
                 dictionary=None,
                 embeddings=None,
                 system_name=None
                 ):
        self.system_name = system_name
        self.dictionary = dictionary
        self.embeddings = embeddings
        self.weight = None
        self.dim = self.embeddings.dim
        self.device = utils.get_device()

    def get_unk(self):
        return self.embeddings.unk

    def get_unk_id(self):
        return self.dictionary.unk_id

    def projection(self, out_dim):
        emb_size = self.embeddings.dim
        assert emb_size >= out_dim
        pca = PCA(n_components=out_dim)

        self.embeddings.V.weight = torch.nn.Parameter(torch.tensor(
            pca.fit_transform(self.embeddings.V.weight.detach().numpy()),
            device=self.device), requires_grad=False)
        self.embeddings.dim = out_dim
        self.dim = out_dim

    def has_w(self, w):
        return w in self.dictionary.tokens

    def get_myid_from_token(self, w):
        myid = self.dictionary.get_myid_from_token(w)

        if myid is None:
            # oov
            return None
        else:
            return myid

    def get_vec_from_token(self, w):
        myid = self.get_myid_from_token(w)
        if myid is None:
            return self.embeddings.unk
        else:
            return self.embeddings.get_vec_from_myid(myid)

    def get_myid_from_wpid(self, wpid):
        return self.dictionary.get_myid_from_wpid(wpid)

    def get_vec_by_wpid(self, wpid):
        myid = self.get_myid_from_wpid(wpid)

        if myid == len(self.dictionary):
            return self.embeddings.unk

        return self.embeddings.V(myid)

    def get_token_from_wpid(self, wpid):
        return self.dictionary.get_token_from_wpid(wpid)

    def get_token_from_myid(self, myid):
        return self.dictionary.get_token_from_myid(myid)

    def get_indices(self, wp_indices):
        """
        slice of my ids

        :param wp_indices: a indexing tensor to retrive the my ids
        :return: my_ids as a tensor
        """
        return self.dictionary.get_indices(wp_indices)


class WordPrism:
    """
    word prism wrapper contains the embeddings
    """

    def __init__(self,
                 num_facets=None,
                 wrappers=None,
                 dim=None
                 ):
        """

        :param num_facets: number of wrappers
        :param wrappers: a list of EmbWrappers wrapping embeddings and
        corresponding dictionaries
        :param dim: the dimension we want to unify all embedding systems
        """
        self.num_facets = num_facets
        self.dim = dim
        # self.weights = nn.init.xavier_uniform_((dim,1)) # ???
        self.embedding_wrappers = wrappers
        self.device = utils.get_device()
        # assert len(self.weights) == num_facets
        assert len(self.embedding_wrappers) == num_facets
        # project all embeddings systems
        need_proj = self.check_projection()
        if need_proj:
            self.projection()

        # produce embedding out of embedding facets
        self.dictionary = self.generate_wp_dictionary()

        # set wpid_myid for each embeddings wrapper
        self.set_wpid_myid()
        self.dataset_vocab_map = None

        self.unk = torch.zeros([self.num_facets, self.dim]).unsqueeze(1)

    def set_dataset_mapping(self, dataset_vocab):
        mapping = []
        cnt = 0
        for token in dataset_vocab.itos:
            if token == "<pad>":
                mapping.append(self.dictionary.padding_id)
            else:
                mapping.append(self.get_id(token))
            if self.get_id(token) != self.dictionary.unk_id:
                cnt += 1
        print("Coverage is {} ({}/{})".format(cnt / len(dataset_vocab), cnt, len(dataset_vocab)))
        self.dataset_vocab_map = torch.LongTensor(mapping).to(self.device)


    def check_projection(self):
        proj = False
        for emb_wrap in self.embedding_wrappers:
            if self.dim != emb_wrap.dim:
                print("need projection !!! ")
                proj = True
                break
        return proj

    def projection(self):
        """
        project all embedding systems to a certain dimension using PCA
        :return: None
        """

        for emb_wrapper in self.embedding_wrappers:
            if emb_wrapper.dim != self.dim:
                emb_wrapper.projection(self.dim)

    def get_id(self, tok):
        return self.dictionary.get_myid_from_token(tok)

    def generate_wp_dictionary(self):
        new_vocab = OrderedDict()
        for emb_wrapper in self.embedding_wrappers:
            new_vocab.update(
                OrderedDict.fromkeys(emb_wrapper.dictionary.tokens))
        dictionary = Dictionary(new_vocab.keys())

        wpid_myid = {}
        for i in range(len(new_vocab)):
            wpid_myid[i] = i
        dictionary.wpid_myid = wpid_myid

        return dictionary

    def set_wpid_myid(self):
        # get the mapper from word facet embedding to word prism embedding
        for i in range(len(self.embedding_wrappers)):
            self.embedding_wrappers[i].dictionary.link_word_prism_id(
                self.dictionary)
            # device=self.device)

    def __len__(self):
        return self.num_facets


class EmbeddingModel(nn.Module):
    """
    A similar class to the nn.Embedding which is a embedding id mapper, from
    token to id map.

    """

    def __init__(self, word_prism, normalize=False,
                 projection=False, proj_dim=-1, projection_normalization=False):
        super(EmbeddingModel, self).__init__()
        _dim = [word_prism.num_facets, word_prism.dim]
        _n_embs = len(word_prism.dictionary) + 2
        self.wp = word_prism


        self.unk_id = len(word_prism.dictionary)
        self.padding_id = len(word_prism.dictionary) + 1
        self.device = utils.get_device()
        self.embedding = self.mapping_index
        self.normalize = normalize
        self.emb_dim = proj_dim if projection else word_prism.dim
        self.projection = projection
        if projection:
            self.projectors = nn.ModuleDict()
            for facet in self.wp.embedding_wrappers:
                self.projectors.update({facet.system_name:
                    nn.Linear(facet.dim,self.emb_dim).to(
                    self.device)})
        self.projection_normalization = projection_normalization

    def mapping_index(self, indices):
        """
        Take the word prism indices and return indices from all facets
        :param indices: shape of (batch_size, sequence_length)
        :return: batch tensor
        """
        # assume all sequence are of the same length
        assert len(indices.shape) == 2
        sequence_len = len(indices[0])
        batch_size = len(indices)
        out_indices = torch.empty((self.wp.num_facets, batch_size,
                                   sequence_len), dtype=torch.long,
                                  device=self.device)
        out_tensor = torch.empty((self.wp.num_facets, batch_size,
                                  sequence_len, self.emb_dim),
                                 device=self.device)

        for i, embwrap in enumerate(self.wp.embedding_wrappers):
            # for facets
            facet_idx = embwrap.get_indices(indices).to(self.device)

            out_indices[i, :] = facet_idx
            # import pdb; pdb.set_trace()

            #  (batch_size, sequence_len, wp_dim)

            if self.projection:
                temp = self.projectors[embwrap.system_name](
                    embwrap.embeddings.V(facet_idx))
                if self.projection_normalization:  # projection followed by normalization
                    temp = utils.normalize(temp, axis=-1) # normalize except the unk embedding
                out_tensor[i, :] = temp
            else:
                out_tensor[i, :] = embwrap.embeddings.V(facet_idx)

        return out_tensor, out_indices

    def forward(self, token_seqs):
        return self.embedding(token_seqs)

    def save_model(self):
        torch.save(self.state_dict(), 'projection.pt')

    def orthogonalize(self, beta=0.01):
        #Same method to enforce orthogonality as in Facebook AI's MUSE (Conneau et al., 2017)
        #Their implementation: https://github.com/facebookresearch/MUSE
        for proj in self.projectors:
            weight = self.projectors[proj].weight.data
            weight.copy_((1 + beta) * weight - beta * weight.mm(weight.transpose(0, 1).mm(weight)))
