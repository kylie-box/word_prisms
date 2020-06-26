# from torch.utils import data
import utils
from nltk.corpus import stopwords
import torchtext
from torchtext import data
import random
import os

STOPS = set(stopwords.words('english'))


class TextClassificationLoader(data.TabularDataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, path, train=None, validation=None,
               test=None
               ):
        fields = {'label': ('label', label_field),
                  'text': ('text', text_field)}
        return super(TextClassificationLoader, cls).splits(path, path,
                                                           "sentiment-train.jsonl",
                                                           "sentiment-dev.jsonl",
                                                           "sentiment-test.jsonl",
                                                           format='json',
                                                           fields=fields)


class TextClassificationIter(object):
    def __init__(self, task=None, wp_dict=None, batch_size=64, root_dir=None):
        dataset = os.path.join(root_dir, 'sup_datasets', task)
        filter_words = get_preprocessing(wp_dict)
        text_field = data.Field(include_lengths=True, init_token='<s>',
                                eos_token='</s>', preprocessing=filter_words,
                                batch_first=True)
        label_field = data.Field(sequential=False, unk_token=None)

        train, dev, test = TextClassificationLoader.splits(text_field,
                                                           label_field,
                                                           dataset)
        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_sizes=(batch_size, 512, 512),
            sort_key=lambda x: len(x.text), sort_within_batch=True,
            repeat=False,
            device=utils.get_device())

        text_field.build_vocab(train, dev, test, min_freq=1)
        label_field.build_vocab(train, dev, test, min_freq=1)
        self.train_iter.shuffle = True
        self.dev_iter.shuffle = False
        self.test_iter.shuffle = False

        self.text_field = text_field
        self.label_field = label_field


def get_emb_key(word, wp_dict):
    ret = word
    try:
        wp_dict.token_ids[word]
    except KeyError:
        try:
            wp_dict.token_ids[word.lower()]
        except KeyError:
            return None
        ret = word.lower()
    return ret


def get_preprocessing(wp_dict):
    def filter_by_emb_vocab(x):
        return [w.strip() for w in x if get_emb_key(w.strip(), wp_dict) is not None]
    return filter_by_emb_vocab


class NLIDatasetLoader(data.TabularDataset):
    dirname = ''
    name = ''
    n_classes = 3

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, fields, root, train='', validation='', test=''):
        return super(NLIDatasetLoader, cls).splits(root, root, train, validation, test, format='json', fields=fields,
                                             filter_pred=lambda ex: ex.label != '-')

    @classmethod
    def get_fields(cls, text_field, label_field, with_genre=False):
        fields = {'label': ('label', label_field), 'sentence1': ('premise', text_field),
                  'sentence2': ('hypothesis', text_field)}
        if with_genre:
            fields['genre'] = ('genre', data.RawField())
        return fields


class SNLIDataset(NLIDatasetLoader):
    dirname = 'snli_1.0'
    name = 'snli'

    @classmethod
    def splits(cls, text_field, label_field, root, train='', validation='', test=''):
        fields = NLIDatasetLoader.get_fields(text_field, label_field, with_genre=False)
        return super(SNLIDataset, cls).splits(fields, root=root, train='snli_1.0_train.tokenized.jsonl',
                                              validation='snli_1.0_dev.tokenized.jsonl',
                                              test='snli_1.0_test.tokenized.jsonl')



class SNLIIter(object):
    def __init__(self, task=None, wp_dict=None, batch_size=64, root_dir=None):
        dataset = os.path.join(root_dir, 'sup_datasets', task)
        filter_words = get_preprocessing(wp_dict)
        text_field = data.Field(include_lengths=True, init_token='<s>',
                                eos_token='</s>', preprocessing=filter_words,
                                batch_first=True)
        label_field = data.Field(sequential=False, unk_token=None)
        train, dev, test = SNLIDataset.splits(text_field, label_field, dataset)

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_sizes=(batch_size, 512, 512),
             sort_within_batch=True,
            repeat=False,
            device=utils.get_device())

        text_field.build_vocab(train, dev, test, min_freq=1)
        label_field.build_vocab(train, dev, test, min_freq=1)
        self.train_iter.shuffle = True
        self.dev_iter.shuffle = False
        self.test_iter.shuffle = False

        self.text_field = text_field
        self.label_field = label_field



