import argparse
from collections import defaultdict
import config
import constants
import numpy as np
import os
import random
import re
import torch
import utils
from embedding_models import EmbeddingModel
from evaluation import dataset_load
from evaluation.results import ResultsHolder
from evaluation.seq_batch_loader import SequenceLoader
from evaluation.train_classifier import sort_by_length, feed_full_ds
from evaluation.train_seq_labeller import feed_full_seq_ds
from evaluation_models import TextClassificationModel, SeqLabModel
from load_as_word_prism import load_word_prism
from logger import create_logger
from sklearn.model_selection import train_test_split
from evaluation.dataset_load import HilbertDataset
from utils import count_param_num


MAX_MB_SIZE = 512


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_optimizer_scheduler(args, nn_module):
    # learning rate scheduler to maximize the validation set accuracy.
    # default with a dummy scheduler where no change occurs

    nn_params = [p for p in nn_module.parameters() if p.requires_grad]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(nn_params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_shrink, patience=args.early_stop_patience // 5,
            mode='max',
            min_lr=args.lr_min, verbose=True)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(nn_params, lr=args.lr, weight_decay=0.99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=args.early_stop_patience // 5,
            mode='max',
            min_lr=args.lr_min, verbose=True)

    elif args.optimizer == 'sgd-m':
        optimizer = torch.optim.SGD(nn_params, lr=args.lr, weight_decay=0.99,
                                    momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=args.early_stop_patience // 5,
            mode='max',
            min_lr=args.lr_min, verbose=True)

    else:
        raise NotImplementedError('Optimizer scheduler \"{}\" not '
                                  'implemented!'.format(args.optimizer))
    return optimizer, scheduler


def data_loading_prep(tr_x, tr_y, te_x, te_y, emb_model, mb_size,
                      seq_labelling=False, val_x=None, val_y=None):
    if val_x is None and val_y is None:
        tr_x, val_x, tr_y, val_y = train_test_split(
            tr_x, tr_y, test_size=0.1, random_state=1848,
        )
    # sort the datasets by length of the sentences, very useful
    tr_x, tr_y = sort_by_length(tr_x, tr_y, reverse=True)
    val_x, val_y = sort_by_length(val_x, val_y, reverse=True)
    te_x, te_y = sort_by_length(te_x, te_y, reverse=True)

    loaders = {'train': None, 'val': None, 'test': None}
    # push everything onto the GPU (hopefully it all fits!)
    loaders['train'] = SequenceLoader(tr_x, tr_y, mb_size,
                                      emb_model.padding_id,
                                      seq_labelling=seq_labelling)
    loaders['val'] = SequenceLoader(val_x, val_y, MAX_MB_SIZE,
                                    emb_model.padding_id,
                                    seq_labelling=seq_labelling)
    loaders['test'] = SequenceLoader(te_x, te_y, MAX_MB_SIZE,
                                     emb_model.padding_id,
                                     seq_labelling=seq_labelling)

    return loaders


def evaluate(model, loaders, args, sst_labels=None):
    train_acc, train_pred_labels, train_gold_labels = feed_full_seq_ds(model, loaders['train'], sst_labels)
    val_acc, val_pred_labels, val_gold_labels = feed_full_seq_ds(model, loaders['val'], sst_labels)
    test_acc, test_pred_labels, test_gold_labels= feed_full_seq_ds(model, loaders['test'], sst_labels)

    return (train_acc, train_pred_labels, train_gold_labels), \
           (val_acc, val_pred_labels, val_gold_labels), \
           (test_acc, test_pred_labels, test_gold_labels)


def train(model, loaders, logger, args, sst_labels=None):
    # determine if we are doing complete batch training
    full_batch_train = args.mb_size == -1
    if full_batch_train:
        args.mb_size = MAX_MB_SIZE  # large minibatches to go fast (but do not exceed the GPU memory)

    single_emb = not (len(
        args.embedding.exp_embs) > 1 or 'all' in args.embedding.exp_embs)
    # get the big daddy bois
    optimizer, scheduler = get_optimizer_scheduler(args, model)

    results = defaultdict(lambda: [])
    best_val_acc = 0
    best_epoch = 0
    early_stop_count = args.early_stop_patience  # if performance doesn't improve for 10 epochs, end it

    system_names = [wrap.system_name for wrap in
                    model.encoder.wp.embedding_wrappers]

    logger.info('model size: {:,}'.format(count_param_num(model)))

    beta = args.evaluation.beta

    if args.verbose: logger.info('Beginning training...')

    # now iterate over the epochs
    for e in range(args.epoch):
        if args.verbose: logger.info('\nEpoch {}: (training)'.format(e))

        # training set iteration
        model.train()
        total = 0
        training_loss = 0
        optimizer.zero_grad()

        # iterate over token sequences and the classification labels for each
        for tok_seqs, tok_pads, label in loaders['train']:

            # check if we are doing full batch trianing, if not, zero-out gradient.
            if not full_batch_train:
                optimizer.zero_grad()
            # make the predictions, compute loss and record it
            predictions = model(tok_seqs, tok_pads)
            loss = model.loss(predictions, label)
            # if loss has diverged just end the misery!!
            if torch.isnan(loss):
                logger.info('DIVERGED!!! ENDING.')
                results.update({'DIVERGED': True})
                hresults = ResultsHolder(args.evaluation.task)
                hresults.add_ds_results('full', results)
                return hresults

            training_loss += loss.data.item()
            total += 1

            # compute the back gradient
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)

            # if we are not doing full batch training, step the optimizer
            if not full_batch_train:
                optimizer.step()
                if args.evaluation.orthogonal and args.embedding.projection:
                    model.encoder.orthogonalize(beta)

        # if we are doing full batch training, we do one big step at the end.
        if full_batch_train:
            optimizer.step()
            if args.evaluation.orthogonal:
                model.encoder.orthogonalize(beta)


        # even out the loss and record it

        if args.task_type == 'sequence_labelling':
            training_loss /= total
        else:
            training_loss /= len(loaders['train'])

        results['loss'].append(training_loss)

        if args.cached and ((e % args.checkpoint_frequency == 0 and e > 0)
                            or e == args.epoch):
            logger.info("saved models at: {}".format(os.path.join(
                args.output_dir, "epoch_{}.checkpoint".format(e))))
            torch.save(model, os.path.join(
                args.output_dir, "epoch_{}.checkpoint".format(e)))
        if not single_emb and not args.evaluation.concat_baseline:
            logger.info('mean {} -- {}'.format(system_names,
                                                          model.saved_weights.tolist()))
            monitor_word_wp_id = model.encoder.wp.get_id(args.evaluation.monitor_word)
            if not args.evaluation.cdme:
                logger.info('monitoring facet-level importance weights\n -{}- : {} -- {}'.format(args.evaluation.monitor_word, system_names,
                                                                                     model.get_weights(torch.LongTensor([[monitor_word_wp_id]])).tolist()
                                                                                     ))

        # now feed forward and get preds for validation set
        if args.verbose: logger.info('    (evaluating...)')
        with torch.no_grad():
            model.eval()
            # import pdb; pdb.set_trace()valid

            (train_acc, _, _), (
            val_acc, val_pred_labels, val_gold_labels), (
            test_acc, test_pred_labels, test_gold_labels) = evaluate(model,
                                                                     loaders,
                                                                     args,
                                                                     sst_labels=sst_labels)


            for acc, string in zip([train_acc, val_acc, test_acc],
                                   ['train', 'val', 'test']):
                results['{}_acc'.format(string)].append(acc)

            # check if it is time to end it
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_train_acc = train_acc
                early_stop_count = args.early_stop_patience
                if e > 10:
                    model.encoder.save_model()
                best_val_pred_labels = val_pred_labels
                best_test_pred_labels = test_pred_labels
            else:
                early_stop_count -= 1
                if early_stop_count <= 0:
                    break

            # print results
            if args.verbose:
                for key in sorted(results.keys()):
                    logger.info(
                        '    {:10} - {:4f}'.format(key, results[key][-1]))

            #### Update the LR schedule! ####
            scheduler.step(val_acc)

    # return the results!
    del loaders['train'], loaders['val']
    results.update({'best_val_acc': best_val_acc,
                    'best_train_acc': best_train_acc,
                    'best_epoch': best_epoch,
                    'test_acc_at_best_epoch': results['test_acc'][best_epoch]})
    hresults = ResultsHolder(args.evaluation.task)
    hresults.add_ds_results('full', results)
    if not os.path.exists("projection.pt"): model.encoder.save_model() # in case of experiments with < 10 epochs
    write_prediction_files(best_val_pred_labels, args.evaluation.id2label, test=False)
    write_prediction_files(best_test_pred_labels, args.evaluation.id2label, test=True)
    write_prediction_files(val_gold_labels, args.evaluation.id2label,
                           test=False, gold=True)
    write_prediction_files(test_gold_labels, args.evaluation.id2label,
                           test=True, gold=True)

    return hresults


def write_prediction_files(predictions, id2label, test=False,gold=False):
    if gold:
        file_name = "test_gold.txt" if test else "val_gold.txt"
    else:
        file_name = "test_prediction.txt" if test else "val_prediction.txt"

    with open(file_name, 'w') as f:
        for l in predictions:
            f.write(id2label[l])
            f.write('\n')


def get_dataset(eval_args, logger):
    logger.info("loading task.. {}".format(eval_args.task))
    loader_dict = {'sst2': dataset_load.load_sst2,
                   'sst-imdb': dataset_load.load_sentiment,
                   'semcor-sst': dataset_load.load_semcor_sst_tagging,
                   'wsj-pos': dataset_load.load_wsj_pos_tagging,
                   'brown-pos': dataset_load.load_brown_pos_tagging,
                   'news': dataset_load.load_news_classification,
                   'conll-pos': dataset_load.load_conll_pos_tagging,
                   'conll-ner': dataset_load.load_conll_ner_tagging,
                   'conll-chunk': dataset_load.load_conll_chunking
                   }
    return loader_dict[eval_args.task](eval_args.dataset_dir)

def get_model(eval_args):
    assert eval_args.task in constants.SEQ_LABELLING
    return SeqLabModel


def get_parser():
    source_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(
        description="Ensemble word embedding project")
    default_config = os.path.join(source_path, "config/master.yaml")
    parser.add_argument("--config", "-c", default=default_config)
    return parser


def main(args):
    # instantiate
    set_seeds(args)

    args.task_type = 'classification' if (
            (args.evaluation.task in constants.ALL_DS) and (
            args.evaluation.task not in constants.SEQ_LABELLING)) else 'sequence_labelling'

    logger = create_logger("log.txt", args.write_logger)
    logger.info('============ Initialized logger ============')
    logger.info('using device {}'.format(utils.get_device()))
    if args.write_logger:
        logger.info("saved models at: {}".format(args.output_dir))
    arg_str = report_args(args)
    logger.info(arg_str)

    # embedding model prep
    if args.embedding.truncation == -1:
        args.embedding.truncation = None
    wp = load_word_prism(args.embedding, logger)


    # dataset prep
    h_dataset = get_dataset(args.evaluation, logger)

    translate_label_by_one = args.task_type == 'sequence_labelling'
    tr_x, tr_y = h_dataset.get_x_y('train', wp.dictionary, as_indicies=True,
                                   translate_label_by_one=translate_label_by_one)
    split = True
    # load from pickled datasets(wsj, brown pos) that do not have function check_valid_exits
    check_valid_exists = getattr(h_dataset, "check_valid_exists", None)
    if callable(check_valid_exists):
        split = not h_dataset.check_valid_exists()

    if split:
        tr_x, val_x, tr_y, val_y = train_test_split(
            tr_x, tr_y, test_size=0.1, random_state=1848,
        )
    else:
        val_x, val_y = h_dataset.get_x_y("val", wp.dictionary, as_indicies=True,
                                         translate_label_by_one=translate_label_by_one)


    te_x, te_y = h_dataset.get_x_y('test', wp.dictionary, as_indicies=True,
                                   translate_label_by_one=translate_label_by_one)

    args.evaluation.n_classes = len(h_dataset.labels_to_idx)
    logger.info('task has {} classes.'.format(args.evaluation.n_classes))

    # models
    if args.evaluation.orthogonal: args.embedding.proj_dim = wp.dim
    emb_model = EmbeddingModel(wp,
                               normalize=args.embedding.normalize,
                               projection=args.embedding.projection,
                               proj_dim=args.embedding.proj_dim,
                               projection_normalization=args.embedding.projection_normalization
                               ).to('cpu')
    model_str = get_model(args.evaluation)
    model = model_str(args.evaluation, emb_model).to(utils.get_device())

    seq_labelling = (args.task_type == 'sequence_labelling')
    loaders = data_loading_prep(tr_x, tr_y, te_x, te_y,
                                emb_model,
                                args.mb_size,
                                seq_labelling=seq_labelling,
                                val_x=val_x,
                                val_y=val_y)

    args.evaluation.id2label = h_dataset.id2labels
    if not args.checkpoint == "None":
        raise NotImplementedError("didn't implement.. oops")

    else:
        try:
            sst_labels = h_dataset.sst_labels

        except AttributeError:
            sst_labels = None


        results = train(model, loaders, logger, args,
                        sst_labels=sst_labels)

    results.serialize(os.path.join(args.output_dir), arg_str)


def report_args(args):
    res = "\n"
    for arg in vars(args):
        res += "{}:\t{}\n".format(arg, getattr(args, arg))
    return res


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    config.create_directory(cfg)
    main(cfg)
