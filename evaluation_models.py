import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import logging
from utils import nn_init, init_weight

logger = logging.getLogger()


class ConcatBaseline(nn.Module):
    def __init__(self, num_facets):
        super(ConcatBaseline, self).__init__()
        self.device = utils.get_device()
        self.num_facets = num_facets
        self.saved_weights = None

    def forward(self, facets):
        res = torch.cat([facets[i,:,:,:] for i in range(self.num_facets)], dim=2)
        return res

class SysLvlLC(nn.Module):
    def __init__(self, num_facets, uniform=True, parameterization_type=None, requires_grad=True):
        super(SysLvlLC, self).__init__()
        self.device = utils.get_device()
        if uniform:
            self.weights = nn.Parameter(
                torch.ones(1, num_facets, device=self.device)
                / torch.ones(1, num_facets, device=self.device).sum(dim=1), requires_grad=requires_grad)
        else:
            self.weights = nn.Parameter(
                torch.empty(1, num_facets, device=self.device), requires_grad=requires_grad)
            init_weight(self.weights)

        self.softmax = nn.Softmax(dim=1)
        self.saved_weights = None
        self.parameterization_type = parameterization_type

    def forward(self, facets):
        """
        linear combination of facets
        :param facets: #of facets * bsz * seq_len * emb_dim
        :return: word embedding of a word
        """
        # number of facets must match
        assert facets.shape[0] == self.weights.shape[-1]
        facets = facets.permute(1,2,3,0)
        res = None
        if self.parameterization_type == 'square':
            res = self.softmax((self.weights)**2).expand_as(facets) * facets
            self.saved_weights = self.softmax(self.weights**2).data
        elif self.parameterization_type == 'raw':
            res = self.softmax(self.weights).expand_as(facets) * facets
            self.saved_weights = self.softmax(self.weights).data
        elif self.parameterization_type is None:
            res = self.weights.expand_as(facets) * facets
            self.saved_weights = self.weights.data
        res = res.sum(-1)

        return res

    def loss(self, loss_type='L1'):
       if loss_type == 'L1':
            return torch.abs(self.weights).sum(dim=1)

    def get_weights(self,words_embs):
        # the order of the facets is fixed according to the emb_wrap order.
        # the output embeddings are looked up as the order they are created.
        return self.saved_weights


class WordLvlLC(nn.Module):
    def __init__(self, emb_dim, emb_dropout=0.0, nonlinear=False, cdme=False):
        super(WordLvlLC, self).__init__()
        self.device = utils.get_device()
        self.cdme = cdme
        if cdme:
            self.attn0 = nn.LSTM(
                    input_size=emb_dim,
                    hidden_size=2,
                    batch_first=True,
                    bidirectional=True)
            self.net = nn.Linear(4, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(emb_dim, 2),
                nn.Linear(2, 1)
            )

        nn_init(self.net, 'xavier')

        self.attn_weights = None
        self.saved_weights = None
        self.dropout = nn.Dropout(p=emb_dropout)
        self.nonlinear = nonlinear

    def get_weights(self, word_embs):
        assert len(word_embs.shape) == 4
        output = F.softmax(self.net(word_embs).squeeze(3), dim=0)
        assert len(output.shape) == 3
        return output.squeeze()


    def forward(self, facets):
        """
        linear layer processing of facets
        :param facets: #of facets * bsz * seq_len * emb_dim
        :return: word embedding of a word
        """

        num_facets, bsz, seq_len, emb_dim = facets.shape
        # to accommodate the linear function, bsz * seq * dim * num_facets
        # facets = facets.view(bsz, seq_len, emb_dim, num_facets)
        if self.cdme:
            # todo: check the CDME results
            rnn_input = facets.view(bsz, seq_len * num_facets, -1)
            self.attn_weights =self.net(self.attn0(rnn_input)[0])
            self.attn_weights = self.attn_weights.view(num_facets, bsz, seq_len,
                                                       1)
            self.attn_weights = F.softmax(self.attn_weights, dim=0)
        else:
            self.attn_weights = self.net(facets)  # facets, bsz, seq_len
            self.attn_weights = F.softmax(self.attn_weights, dim=0)
        res = facets * self.attn_weights.expand_as(facets)
        res = res.sum(0)
        # self.saved_weights = (self.attn0.weight.flatten(), self.attn1.weight.flatten())
        if len(res.size()) == 3:
            assert res.shape[0] == bsz and res.shape[1] == seq_len and \
                   res.shape[2] == emb_dim
        else:
            assert res.shape[0] == seq_len and res.shape[1] == emb_dim
        if self.nonlinear:
            res = F.relu(res)
        if self.dropout.p> 0:
            res = self.dropout(res)
        self.saved_weights = self.attn_weights.mean(dim=1).mean(dim=1)
        return res


class ContextualizedLvlLC(nn.Module):
    def __init__(self, hidden_dim, emb_dropout, sequence_labelling=False):
        super(ContextualizedLvlLC, self).__init__()
        self.sequence_labelling = sequence_labelling
        self.device = utils.get_device()
        self.net = nn.Linear(hidden_dim, 1)
        init_weight(self.net.weight)
        self.attn_weights = None
        self.saved_weights = None
        self.dropout = nn.Dropout(p=emb_dropout)

    def get_weights(self, hstates):
        """
        a particular sequence!
        :param hstates:
        :return:
        """
        # assert len(word_embs.shape) == 4
        output = self.net(hstates).squeeze(
                2)  # facets, bsz
        output = F.softmax(output, dim=0)
        assert len(output.shape) == 3
        return output

    def forward(self, hstates):
        """
        linear layer processing of facets
        :param hstates: #of facets * bsz * seq_len * rnn_hstate_dim
        :return: word embedding of a word
        """
        if self.sequence_labelling:
            num_facets, bsz, seq_len, rnn_hstate_dim = hstates.shape
            self.attn_weights = self.net(hstates).squeeze(
                3)  # facets, bsz, seq_len
            self.attn_weights = F.softmax(self.attn_weights, dim=0)
            res = hstates * self.attn_weights.view(num_facets, bsz, seq_len,
                                                   1).expand_as(hstates)
            res = res.sum(0)
            assert res.shape[0] == bsz and res.shape[1] == seq_len and \
                   res.shape[2] == rnn_hstate_dim
        else:
            num_facets, bsz, rnn_hstate_dim = hstates.shape
            self.attn_weights = self.net(hstates).squeeze(
                2)  # facets, bsz
            self.attn_weights = F.softmax(self.attn_weights, dim=0)
            res = hstates * self.attn_weights.view(num_facets, bsz,
                                                   1).expand_as(hstates)
            res = res.sum(0)
            assert res.shape[0] == bsz and res.shape[1] == rnn_hstate_dim
            res = F.relu(res)
            self.saved_weights = self.attn_weights.mean(dim=1)
        if self.dropout.p > 0:
            res = self.dropout(res)
        return res


class BaseEvaluationModel(nn.Module):
    def __init__(self, eval_args, word_prism_model):
        super(BaseEvaluationModel, self).__init__()
        self.args = eval_args
        self.device = utils.get_device()
        self.encoder = word_prism_model
        self.hidden_dim = eval_args.rnn_dim
        self.meta_embs = len(self.encoder.wp.embedding_wrappers) > 1
        # self.lc = WordLvlLC(word_prism_model.emb_dim, 0.5) \
        #     if self.meta_embs and self.args.prism_level != "contextualized" else None
        self.saved_weights = None # store numerical value of the weights after softmax
        if eval_args.average_baseline: eval_args.parameterization_type = None
        self.lc = self.get_lc(eval_args.prism_level, num_facets=self.encoder.wp.num_facets,
                              emb_dim=self.encoder.emb_dim, concat=eval_args.concat_baseline,
                              requires_grad=(not eval_args.average_baseline),
                              parameterization_type=eval_args.parameterization_type,
                              cdme=eval_args.cdme)
        self.prism_level = self.args.prism_level
        self.device = utils.get_device()
        self.lstm = nn.ModuleDict()
        self.system_names = []
        if eval_args.lstm_layers == 1:
            eval_args.rnn_dropout = 0

        # contextualized
        if self.meta_embs and self.prism_level == "contextualized":
            for emb_wrapper in self.encoder.wp.embedding_wrappers:
                self.system_names.append(emb_wrapper.system_name)
                # we got F lstms! The number of parameter is not good..
                self.lstm.update({emb_wrapper.system_name:
                    nn.LSTM(
                        input_size=word_prism_model.emb_dim,
                        hidden_size=eval_args.rnn_dim,    # 64
                        num_layers=eval_args.lstm_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=eval_args.rnn_dropout)})
                nn_init(self.lstm[emb_wrapper.system_name], 'orthogonal')

        else:
            if not self.args.concat_baseline:
                self.lstm.update(
                    {'prism': nn.LSTM(input_size=word_prism_model.emb_dim,
                                      hidden_size=eval_args.rnn_dim,
                                      num_layers=eval_args.lstm_layers,
                                      batch_first=True,
                                      bidirectional=True)})
            else:
                self.lstm.update(
                    {'prism': nn.LSTM(input_size=word_prism_model.emb_dim * self.encoder.wp.num_facets,
                                      hidden_size=eval_args.rnn_dim,
                                      num_layers=eval_args.lstm_layers,
                                      batch_first=True,
                                      bidirectional=True)})
            nn_init(self.lstm['prism'], 'orthogonal')

        self.hidden = None
        self.n_layers = eval_args.lstm_layers
        self.lc_weights_loss = None if eval_args.lc_weights_loss in['None', 'none'] \
                            else eval_args.lc_weights_loss

        if self.lc_weights_loss is not None:
            assert self.lc_weights_loss in ['L1']
            self.loss_lambda = eval_args.loss_lambda

        self.proj_normalization = eval_args.proj_normalization
        if self.proj_normalization:
            self.proj_lambda = eval_args.proj_lambda

    def get_lc(self, prism_level="system", num_facets=0, emb_dim=256,
               emb_dropout=0.5, concat=False, requires_grad=True,
               parameterization_type=None, cdme=False):
        if not concat:
            if parameterization_type == 'None': parameterization_type = None
            if parameterization_type is not None:
                assert parameterization_type in ['raw', 'square']
                assert requires_grad == True
            assert prism_level in ["system", "word", "contextualized"]
            if prism_level == "system":
                return SysLvlLC(num_facets, parameterization_type=parameterization_type,
                                requires_grad=requires_grad)
            elif prism_level == "word":
                return WordLvlLC(emb_dim, emb_dropout, cdme=cdme)
            elif prism_level == "contextualized":
                return ContextualizedLvlLC(self.hidden_dim*2, emb_dropout)
            else:
                raise ValueError("Unknown prism level!")
        else:
            return ConcatBaseline(num_facets)

    def init_hidden(self, bsz, contextualized=False):
        hstate = torch.zeros(self.n_layers * 2, bsz, self.hidden_dim).to(
            self.device)
        cstate = torch.zeros(self.n_layers * 2, bsz, self.hidden_dim).to(
            self.device)
        states = {}
        if contextualized:
            for system_name in self.system_names:
                states[system_name] = (hstate, cstate)
        else:
            states['prism'] = (hstate, cstate)
        return states

    def encode_sentence(self, emb_seqs, pads, facet_name="prism"):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def get_weights(self, wp_idx):
        """

        :param wp_idx: shape is (1, # of words need to query)
        :return: weights
        """
        assert wp_idx.shape[0] == 1
        assert len(wp_idx.shape) == 2
        emb_seq, _ = self.encoder(wp_idx)
        return self.lc.get_weights(emb_seq)

class TextClassificationModel(BaseEvaluationModel):
    def __init__(self, eval_args, word_prism_model):
        super(TextClassificationModel, self).__init__(eval_args,
                                                      word_prism_model)
        self.classifier = self.get_classifier()
        nn_init(self.classifier, 'xavier')
        self.loss_function = nn.CrossEntropyLoss()
        self.lc.sequence_labelling = False   # for contextualize prisms

    def get_classifier(self):
        if self.args.hilbert:
            # as in Hilbert, ffnn option True
            return nn.Sequential(
                nn.Dropout(p=self.args.clf_dropout),
                nn.Linear(2 * self.args.rnn_dim, self.args.fc_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=self.args.clf_dropout),
                nn.Linear(self.args.fc_dim, self.args.n_classes),
            )
        else:
            # DME version of classifier
            return nn.Sequential(
                nn.Linear(2 * self.args.rnn_dim, self.args.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.args.clf_dropout),
                nn.Linear(self.args.fc_dim, self.args.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.args.clf_dropout),
                nn.Linear(self.args.fc_dim, self.args.n_classes),
            )


    def get_weights(self, wp_idx):
        if self.prism_level == 'contextualized':
            """
            feed forward a sample sentence that contains the word of interest, 
            not necessarily be in one of the training examples. report the distribution
            of weights of the last hidden layer in classification and the weight of the word 
            for sequence labelling.
            """
            weights = None
            example_sent = "the cat sat on the mat"
            self.hidden = self.init_hidden(1, contextualized=True)
            indices = torch.LongTensor(
                [[self.encoder.wp.get_id(tok) for tok in example_sent.split()]])
            assert wp_idx in indices
            emb_seqs, _ = self.encoder(indices)
            res = torch.zeros(emb_seqs.shape[0], emb_seqs.shape[1],
                              2 * self.hidden_dim, device=self.device)
            for i, (f_emb_seq, f_name) in enumerate(
                    zip(emb_seqs, self.system_names)):
                # encode each facet with different lstms
                temp = self.encode_sentence(f_emb_seq, torch.LongTensor([0]),
                                            f_name)
                res[i] = temp
            X = self.lc(
                res)  # X = sigma(hstate1 *(W hstate1) + ... + hstate_n*(W hstate_n)) W is shared

            weights = self.lc.saved_weights
            return weights

        else:
            return super(TextClassificationModel, self).get_weights(wp_idx)

    def encode_sentence(self, emb_seqs, pads, facet_name="prism"):
        """
        get the repr of a sentence of a facet
        :param emb_seqs: padded embedding sequences of shape bsz * max_seq_len * wp dim
        :param pads: a vector of the number of padding token in each batch
        :param facet_name: name of the facet working on for contextualized level, if is "prism" it means there is only one biLSTM
        :return: encoded sentences in the batch
        """
        bsz, max_seq_len, emb_dim = emb_seqs.shape
        prism_packed = nn.utils.rnn.pack_padded_sequence(
            emb_seqs, max_seq_len - pads, batch_first=True)
        X, (hidden_state, cell_state) = self.lstm[facet_name](prism_packed, self.hidden[facet_name])
        X = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)[0]

        if self.args.max_pooling:
            # shape of X is B x L x 2h, so need to max-pool along dim=1, L
            X = torch.max(X, dim=1)[0]
            assert (X.shape == (bsz, self.hidden_dim * 2))
        else:
            last_backward = hidden_state[-1, :, :]
            last_forward = hidden_state[-2, :, :]
            X = torch.cat((last_forward, last_backward), dim=1)
        return X  # bsz, 2h

    def get_sent_repr_lc(self, p_emb_seqs, pads):
        """
        get the lc of sentence representations
        :param p_emb_seqs: num_facets * bsz * max_seq_len * wp dim
        :return: a matrix of batch encoded sentences: bsz * hidden_dim * 2
        """

        encoded_facets = torch.empty(p_emb_seqs.shape[0], p_emb_seqs.shape[1],
                                     self.hidden_dim * 2,
                                     device=self.device)  # num_facets, bsz, 2h
        for i, emb_seqs in enumerate(p_emb_seqs):
            sents = self.encode_sentence(emb_seqs, pads)  # bsz, 2h
            encoded_facets[i] = sents
        res = self.lc(encoded_facets)
        return res


    def forward(self, dataset_id_seq, sequence_lengths):
        """
        encode batch sentences using wp id.

        :param dataset_id_seq: words - dataset vocab id map
        :param sequence_lengths: lengths of each sequence of paddings
        :return: encoded sentence
        """
        wp_id_seq = self.encoder.wp.dataset_vocab_map[dataset_id_seq]
        emb_seqs, emb_toks = self.encoder(wp_id_seq)
        # note: emb_seqs -> (prism_size, batch_size, max_seq_len, embedding_dim)
        # padded.
        # emb_toks -> (prism_size, batch_size, max_seq_len)
        prism_size, bsz, max_seq_len, emb_dim = emb_seqs.shape
        contextualized = True if self.prism_level=="contextualized" else False
        self.hidden = self.init_hidden(bsz, contextualized=contextualized)

        # linear combination of facets, output a single embedding for each word
        if self.meta_embs:
            if self.prism_level in ["system", "word"]:
                assert len(self.lstm) == 1 # only one lstm
                # apply function on static embeddings first
                p_emb_seq = self.lc(emb_seqs)  # bsz, max_seq_len, dim
                X = self.encode_sentence(p_emb_seq, max_seq_len - sequence_lengths)

            elif self.prism_level == 'contextualized':
                # we have # of facets bi-LSTMs, finally we combine them with linear combination before going into the classifer layer
                assert len(self.lstm) > 1

                res = torch.zeros(prism_size, bsz, 2*self.hidden_dim, device=self.device)
                for i, (f_emb_seq, f_name) in enumerate(zip(emb_seqs,self.system_names)):
                    # encode each facet with different lstms
                    temp = self.encode_sentence(f_emb_seq, max_seq_len - sequence_lengths, f_name)
                    assert temp.shape[0] == bsz
                    assert temp.shape[1] == 2*self.hidden_dim
                    res[i] = temp
                X = self.lc(res)    # X = sigma(hstate1 *(W hstate1) + ... + hstate_n*(W hstate_n)) W is shared
                # shape of X is B x L x 2h, so need to max-pool along dim=1, L
                assert (X.shape == (bsz, self.hidden_dim * 2))
            self.saved_weights = self.lc.saved_weights

        else:
            p_emb_seq = emb_seqs[0]  # bsz, max_seq_len, dim
            X = self.encode_sentence(p_emb_seq, max_seq_len - sequence_lengths)

        y = self.classifier(X)
        return F.log_softmax(y, dim=1).squeeze()

    def loss(self, predictions, label):
        loss = self.loss_function(predictions, label)

        if self.lc_weights_loss:
            loss += self.loss_lambda * self.lc.loss(loss_type=self.lc_weights_loss)
        if self.proj_normalization:
            for proj in self.encoder.projectors:
                loss += self.proj_lambda * torch.norm(self.encoder.projectors[proj].weight) / self.encoder.wp.num_facets

        return loss


class NLIModel(BaseEvaluationModel):
    def __init__(self, eval_args, word_prism_model):
        super(NLIModel, self).__init__(eval_args, word_prism_model)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 2 * eval_args.rnn_dim, eval_args.fc_dim),
            nn.ReLU(),
            nn.Dropout(p=eval_args.clf_dropout),
            nn.Linear(eval_args.fc_dim, eval_args.fc_dim),
            nn.ReLU(),
            nn.Dropout(p=eval_args.clf_dropout),
            nn.Linear(eval_args.fc_dim, eval_args.n_classes),
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.lc.sequence_labelling = False  # for contextualize prisms
        if self.prism_level == 'contextualized':
            self.lc.net = nn.Linear(4 * 2 * eval_args.rnn_dim,1)
        self.saved_weights = None


    def encode_sentence(self, emb_seqs, pads, facet_name="prism"):
        """
        get the repr of a sentence of a facet
        :param emb_seqs: padded embedding sequences of shape bsz * max_seq_len * wp dim
        :param pads: a vector of the number of padding token in each batch
        :param facet_name: name of the facet working on for contextualized level, if is "prism" it means there is only one biLSTM
        :return: encoded sentences in the batch
        """
        bsz, max_seq_len, emb_dim = emb_seqs.shape
        prism_packed = nn.utils.rnn.pack_padded_sequence(
            emb_seqs, max_seq_len - pads, batch_first=True)
        X, (hidden_state, cell_state) = self.lstm[facet_name](prism_packed, self.hidden[facet_name])
        X = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)[0]

        if self.args.max_pooling:
            # shape of X is B x L x 2h, so need to max-pool along dim=1, L
            X = torch.max(X, dim=1)[0]
            assert (X.shape == (bsz, self.hidden_dim * 2))
        else:
            last_backward = hidden_state[-1, :, :]
            last_forward = hidden_state[-2, :, :]
            X = torch.cat((last_forward, last_backward), dim=1)
        return X  # bsz, 2h

    def sorted_forward(self, unsorted_embs, unsorted_lengths, max_length, facet_name='prism'):
        seq_len, idx_sorted = torch.sort(unsorted_lengths, descending=True)
        _, idx_unsort = torch.sort(idx_sorted, descending=False)
        sorted_input = unsorted_embs.index_select(0, idx_sorted.cuda())
        sorted_output = self.encode_sentence(sorted_input, max_length - seq_len, facet_name)
        rnn_output = sorted_output.index_select(0, idx_unsort.cuda())
        return rnn_output

    def forward(self, dataset_id_seq, sequence_lengths):
        """
        we encode the premise and hypothesis with [p, h, |h-p|, h*p]
        :param dataset_id_seq: tuple of h and p
        :param sequence_lengths: tuple of lengths of h and p
        :return: class
        """
        hypothesis_len = sequence_lengths[0]
        premise_len = sequence_lengths[1]

        hypothesis = dataset_id_seq[0]
        premise = dataset_id_seq[1]

        h_wp_id_seq = self.encoder.wp.dataset_vocab_map[hypothesis]
        p_wp_id_seq = self.encoder.wp.dataset_vocab_map[premise]
        h_emb_seqs, _ = self.encoder(h_wp_id_seq)
        p_emb_seqs, _ = self.encoder(p_wp_id_seq)

        prism_size, bsz, hypothesis_max_seq_len, emb_dim = h_emb_seqs.shape
        _, _, premise_max_seq_len, _ = p_emb_seqs.shape
        contextualized = True if self.prism_level == "contextualized" else False
        self.hidden = self.init_hidden(bsz, contextualized=contextualized)

        # linear combination of facets, output a single embedding for each word
        if self.meta_embs:
            if self.prism_level in ["system", "word"]:
                assert len(self.lstm) == 1  # only one lstm
                # apply function on static embeddings first
                h_prism_emb_seq = self.lc(h_emb_seqs)  # bsz, max_seq_len, dim
                p_prism_emb_seq = self.lc(p_emb_seqs)
                encoded_hypothesis = self.sorted_forward(h_prism_emb_seq,
                                         hypothesis_len, hypothesis_max_seq_len)

                encoded_premise = self.sorted_forward(p_prism_emb_seq,
                                          premise_len, premise_max_seq_len)
                X = torch.cat([encoded_hypothesis, encoded_premise, (encoded_hypothesis - encoded_premise).abs(), encoded_hypothesis * encoded_premise], dim=1)
            elif self.prism_level == 'contextualized':
                # we have # of facets bi-LSTMs, finally we combine them with linear combination before going into the classifer layer
                assert len(self.lstm) > 1

                emb_seqs = (h_emb_seqs, p_emb_seqs)
                res = torch.zeros(prism_size, bsz, 4* 2 * self.hidden_dim,
                                  device=self.device)
                for i, (f_emb_seq, f_name) in enumerate(
                        zip(emb_seqs, self.system_names)):
                    # encode each facet with different lstms
                    h_temp = self.sorted_forward(f_emb_seq[0],
                                                hypothesis_len, hypothesis_max_seq_len,
                                                f_name)

                    p_temp = self.sorted_forward(f_emb_seq[1],
                                                  premise_len, premise_max_seq_len,
                                                  f_name)
                    assert p_temp.shape[0] == bsz
                    assert h_temp.shape[0] == bsz
                    temp = torch.cat([h_temp, p_temp,
                                   (h_temp - p_temp).abs(),
                                   h_temp * p_temp], dim=1)
                    assert temp.shape[1] == 2 * 4 * self.hidden_dim
                    res[i] = temp
                X = self.lc(
                    res)  # X = sigma(hstate1 *(W hstate1) + ... + hstate_n*(W hstate_n)) W is shared
                # shape of X is B x L x 2h, so need to max-pool along dim=1, L
                assert (X.shape == (bsz, self.hidden_dim * 2 * 4))
            self.saved_weights = self.lc.saved_weights

        else:
            h_prism_emb_seq = h_emb_seqs[0]  # bsz, max_seq_len, dim
            p_prism_emb_seq = p_emb_seqs[0]
            encoded_hypothesis = self.sorted_forward(h_prism_emb_seq,
                                                     hypothesis_len,
                                                     hypothesis_max_seq_len)

            encoded_premise = self.sorted_forward(p_prism_emb_seq,
                                                  premise_len,
                                                  premise_max_seq_len)
            X = torch.cat([encoded_hypothesis, encoded_premise,
                           (encoded_hypothesis - encoded_premise).abs(),
                           encoded_hypothesis * encoded_premise], dim=1)

        y = self.classifier(X)
        return F.log_softmax(y, dim=1).squeeze()

    def loss(self, predictions, label):
        loss = self.loss_function(predictions, label)

        if self.lc_weights_loss:
            loss += self.loss_lambda * self.lc.loss(
                loss_type=self.lc_weights_loss)
        if self.proj_normalization:
            for proj in self.encoder.projectors:
                loss += self.proj_lambda * torch.norm(self.encoder.projectors[
                                                          proj].weight) / self.encoder.wp.num_facets

        return loss


class SeqLabModel(BaseEvaluationModel):
    # universal constant, do not change!
    PADDING_LABEL_ID = 0

    # extends the EmbeddingModel class which uses our pretrained embeddings.
    def __init__(self, eval_args, word_prism_model):
        super(SeqLabModel, self).__init__(eval_args, word_prism_model)
        assert eval_args.rnn_dim > 0 and eval_args.n_classes > 0 and eval_args.lstm_layers > 0
        self.n_labels = eval_args.n_classes
        # output label prediction at each time step
        self.hidden_to_label = nn.Linear(self.hidden_dim * 2, self.n_labels)

        # don't do hidden initialization until we know the batch size
        self.hidden = None
        self.lc.sequence_labelling = True

    def encode_sentence(self, emb_seq, pads, facet_name="prism"):
        bsz, max_seq_len, emb_dim = emb_seq.shape
        X = nn.utils.rnn.pack_padded_sequence(emb_seq,
                                              max_seq_len - pads,
                                              batch_first=True)
        X, self.hidden[facet_name] = self.lstm[facet_name](X, self.hidden[facet_name])
        X = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)[0]
        X = X.contiguous()
        return X

    def get_weights(self, wp_idx):
        if self.prism_level == 'contextualized':
            """
            feed forward a sample sentence that contains the word of interest, 
            not necessarily be in one of the training examples. report the distribution
            of weights of the last hidden layer in classification and the weight of the word 
            for sequence labelling.
            """
            weights = None
            example_sent = "the cat sat on the mat"
            self.hidden = self.init_hidden(1, contextualized=True)
            indices = torch.LongTensor(
                [[self.encoder.wp.get_id(tok) for tok in example_sent.split()]])
            assert wp_idx in indices
            emb_seqs, _ = self.encoder(indices)
            res = torch.zeros(emb_seqs.shape[0], emb_seqs.shape[1], emb_seqs.shape[2], 2 * self.hidden_dim,
                              device=self.device)
            for i, (f_emb_seq, f_name) in enumerate(
                    zip(emb_seqs, self.system_names)):
                # encode each facet with different lstms
                temp = self.encode_sentence(f_emb_seq, torch.LongTensor([0]),
                                            f_name)
                res[i] = temp
            X = self.lc(
                res)  # X = sigma(hstate1 *(W hstate1) + ... + hstate_n*(W hstate_n)) W is shared
            weights = self.lc.attn_weights[:,0,1]
            return weights
        else:
            return super(SeqLabModel, self).get_weights(wp_idx)


    def forward(self, sorted_tok_ids, pads):
        # get the tensor with emb sequences, along with the number of pads in each seq
        emb_seqs, emb_toks = self.encoder(sorted_tok_ids)

        # now we gotta do some special packing
        # note: emb_seqs -> (batch_size, max_seq_len, embedding_dim)
        prism_size, bsz, max_seq_len, emb_dim = emb_seqs.shape
        contextualized = True if self.prism_level=="contextualized" else False
        self.hidden = self.init_hidden(bsz, contextualized=contextualized)

        if self.meta_embs:
            if self.prism_level in ["system", "word"]:  # prism level == system/word
                assert len(self.lstm) == 1
                p_emb_seq = self.lc(emb_seqs)  # bsz, max_seq_len, dim
                X = self.encode_sentence(p_emb_seq, pads)
            else:
                # we have # of facets bi-LSTMs, finally we combine them with linear combination before going into the classifer layer
                assert len(self.lstm) > 1
                res = torch.zeros(prism_size, bsz, max_seq_len, 2 * self.hidden_dim,
                                  device=self.device)
                for i, (f_emb_seq, f_name) in enumerate(
                        zip(emb_seqs, self.system_names)):
                    # encode each facet with different lstms
                    temp = self.encode_sentence(f_emb_seq, pads, facet_name=f_name)
                    assert temp.shape[0] == bsz
                    assert temp.shape[1] == max_seq_len
                    assert temp.shape[2] == 2 * self.hidden_dim
                    res[i] = temp
                # X = sigma(hstate1 *(W hstate1) + ... + hstate_n*(W hstate_n)) W is shared
                X = self.lc(res)
                # shape of X is B x L x 2h, so need to max-pool along dim=1, L
                X = X.contiguous()
            self.saved_weights = self.lc.saved_weights
        else:
            p_emb_seq = emb_seqs[0]  # bsz, max_seq_len, dim
            X = self.encode_sentence(p_emb_seq, pads)

        # run through the linear tag prediction
        X = X.view(-1, X.shape[2])
        X = self.hidden_to_label(X)  # dim is max_len * bsz, n_labels

        # softmax activations in the feed forward for an easy main method
        Y_hat = F.log_softmax(X, dim=1)
        return Y_hat.view(bsz, max_seq_len, self.n_labels)

    def loss(self, Y_hat, padded_labels):
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches...
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels, a longtensor of labels (with padding)
        Y = padded_labels.view(-1)
        mask = (
                Y != self.PADDING_LABEL_ID).float()  # create a mask by zeroing out padding tokens

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.n_labels)

        # count how many tokens we have
        n_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        # must do Y - 1 for the real label indexes (pushed from the padding label, which is 0)
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y - 1] * mask

        # compute cross entropy loss which ignores all the pads
        ce_loss = -torch.sum(Y_hat) / n_tokens

        loss = ce_loss

        if self.lc_weights_loss:
            loss += self.loss_lambda * self.lc.loss(loss_type=self.lc_weights_loss)
        if self.proj_normalization:
            for proj in self.encoder.projectors:
                loss += self.proj_lambda * torch.norm(self.encoder.projectors[proj].weight) / self.encoder.wp.num_facets

        return loss

    def get_label_predictions(self, Y_hat, padded_labels):
        # flatten all the labels, a longtensor of labels (with padding)
        Y = padded_labels.view(-1)
        mask = (Y != self.PADDING_LABEL_ID).long()

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.n_labels)

        # pick the values for the label
        _, preds = torch.max(Y_hat, dim=1)

        # zero out the paddings preds and return the label predictions
        # the plus one is for pushing them back to the label indexes
        return (preds + 1) * mask
