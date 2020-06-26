from nltk.corpus import stopwords

import json
import sys

import os
import torch

# basic
EMB_DIM = 786
STOPS = set(stopwords.words('english'))
SENTI_STOPS = STOPS.copy()
SENTI_STOPS.difference_update({'no', 'not'})
SENTI_STOPS.update({'.'})

# dataset names
SIMILARITY = 'similarity'
ANALOGY = 'analogy'
BROWN_POS = 'brown-pos'
WSJ_POS = 'wsj-pos'
CONLL_POS = 'conll-pos'
CONLL_CHUNKING = 'conll_chunking'
CONLL_NER = 'conll-ner'
SST = 'semcor-sst'
CHUNKING = 'chunking'
IMDB = 'sst-imdb'
SST2 = 'sst2'
NEWS = 'news'
SNLI = 'snli'


ALL_DS = [SIMILARITY, ANALOGY, BROWN_POS, WSJ_POS, CHUNKING, IMDB, NEWS, SST, SST2, CONLL_POS, CONLL_CHUNKING, CONLL_NER]
SUP_DS = [BROWN_POS, WSJ_POS, IMDB, NEWS, CHUNKING, SST, SST2, CONLL_POS, CONLL_CHUNKING, CONLL_NER]

SEQ_LABELLING = [BROWN_POS, WSJ_POS, SST, CONLL_POS, CONLL_CHUNKING, CONLL_NER]



def read_rc():
    RC = {
        'cooccurrence_dir': None,
        'corpus_dir': None,
        'embeddings_dir': None,
        'task_data_path': None,
        'device': 'cuda',
        'dtype': '32',
        'max_sector_size': '12000',
    }
    try:
        with open(os.path.expanduser('~/.hilbertrc')) as rc_file:
            found_rc = json.loads(rc_file.read())
            for key in found_rc:
                RC[key] = found_rc[key]
    except OSError:
        pass

    # Interpret ints
    for int_field in ['max_sector_size']:
        RC[int_field] = int(RC[int_field])

    # Convert dtype specification into an actual torch dtype.
    RC['dtype'] = {
        'half': torch.float16, 'float': torch.float32, 'double': torch.float64,
        '16': torch.float16, '32': torch.float32, '64': torch.float64,
    }[RC['dtype']]

    return RC


RC = read_rc()

MATRIX_DEVICE = 'cuda'  # TODO: purge this global away
MEMORY_DEVICE = 'cpu'  # TODO: purge this global away
DEFAULT_DTYPE = torch.float32
CODE_DIR = os.path.abspath(os.path.join(__file__, '..'))
TEST_DIR = os.path.join(CODE_DIR, 'tests', 'test-data')
TEST_TOKEN_PATH = os.path.join(TEST_DIR, 'test_doc.txt')
TEST_DOCS_DIR = os.path.join(TEST_DIR, 'test-docs')
PAD = sys.maxsize
