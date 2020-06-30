import os

import utils
from embedding_models import WordPrism, EmbWrapper, Embeddings
from logger import create_logger

"""
each directory named as the system contains the pretrained embeddings V.npy and 
dictionary. Load as the embedding wrapper and build the word prism

Note that the word embeddings from cached V.npy are normalized! Don't compare it
with the original embeddings. They are correct!!! 
"""


def load_embwrapper(embeddings=None, dictionary=None, system_name=None):
    return EmbWrapper(dictionary=dictionary, embeddings=embeddings,
                      system_name=system_name)


def load_word_prism(args, logger=None):
    """
    load word prism
    :param args: arugment holder
    :param logger: logger to put information
    :return: word prim: the word prism that contains all embeddings system
    """

    if args.exp_embs[0] == 'all':
        directory = [d for d in os.listdir(args.embeds_root) if
                     not d.startswith('.') and not d.startswith('dme')]
    elif args.exp_embs[0].startswith('random'):
        directory = []
    else:
        directory = [d for d in os.listdir(args.embeds_root) if
                     d in args.exp_embs and not d.startswith(
                         '.') and not d.startswith('dme')]

    device = utils.get_device()
    emb_wraps = []
    # TODO: remove emb wrapper, leave embedding only

    if len(directory) == 0:
        assert args.exp_embs[0].startswith('random')
        random_embs_num = int(args.exp_embs[0][6])
        for random_cnt in range(0, random_embs_num):
            embeddings = Embeddings.load_embedding(
                os.path.join(args.embeds_root, 'win1'), -1, device=device,
                random_cnt=random_cnt)
            dictionary = embeddings.dictionary
            emb_wrap = load_embwrapper(embeddings, dictionary, system_name="random{}".format(random_cnt + 1))
            emb_wraps.append(emb_wrap)
    else:
        for d in directory:

            abs_path = os.path.join(args.embeds_root, d)
            if args.normalize:
                logger.info("normalizing...")
            if d == 'concept_net':
                embeddings = Embeddings.load_embedding(abs_path, -1,
                                                       normalize=args.normalize,
                                                       device=device)
            else:
                embeddings = Embeddings.load_embedding(abs_path, args.truncation,
                                                       normalize=args.normalize,
                                                       device=device)
            dictionary = embeddings.dictionary
            if logger is not None:
                logger.info("{} embeddings of shape {} loaded..".format(d,
                                                                        embeddings.V.weight.shape))
            emb_wrap = load_embwrapper(embeddings, dictionary, system_name=d)
            emb_wraps.append(emb_wrap)

    emb_dims = min([emb_wrap.dim for emb_wrap in emb_wraps])
    wp = WordPrism(num_facets=len(emb_wraps), wrappers=emb_wraps, dim=emb_dims)
    if logger is not None: logger.info(
        "word prism dictionary size {}".format(len(wp.dictionary)))
    return wp

