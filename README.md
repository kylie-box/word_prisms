# Word Prisms

Word embeddings are trained to predict word cooccurrence statistics, which leads them to possess different lexical properties (syntactic, semantic, synonymy, hypernymy, etc.) depending on the notion of context defined at training time.
These properties manifest when querying the embedding space for the most similar vectors, and when used at the input layer of deep neural networks trained to solve NLP problems.
Meta-embeddings combine multiple sets of differently trained word embeddings, and have been shown to successfully improve intrinsic and extrinsic performance over equivalent models which use just one set of source embeddings. 
We introduce *word prisms*: a simple and efficient meta-embedding method that learns to combine source embeddings according to the task at hand. 
Word prisms learn orthogonal transformations to linearly combine the input source embeddings, which allows them to be very efficient at inference time.
We evaluate word prisms in comparison to other meta-embedding methods on six extrinsic evaluations and observe that word prisms offer improvements in performance on all tasks.

## Requirements

* Python 3.6+
* NumPy 1.18.1
* SciPy 1.4.1
* PyTorch 1.5.0
* PyYAML 5.3.1
* EasyDict 1.9
* NLTK 3.5
* scikit-learn 0.23.0
* Hilbert: To install, clone [this](https://github.com/enewe101/hilbert) repository and run `python setup.py develop`
* Hilbert_experiments: To install, clone [this](https://github.com/kylie-box/hilbert_experiments) repository and run `python setup.py develop`

## Source Embeddings

To run word prisms, your source embeddings must be saved in their own directory. This directory must contain an a .npy file called `V.npy`, which stores a numpy array of shape `(vocabulary_size, embedding dimension)`, as well as a dictionary file (simply called `dictionary`) which lists words in the same order as the appear in the rows of the array in `V.npy`.

If your embeddings are in GloVe format (a .txt file where each line consists of a word followed by the entries of its embedding vector, separated by blank spaces), run `python convert_as_word_prism -d [Path to directory containing your embeddings]`.

Now, make a parent directory which contains each of the source embedding directories as a subdirectory. For example, you can have a directory called `facets`, which contains the subdirectories `fasttext` and `glove`.

## Datasets

All datasets would be saved in a single directory named `sup_dataset`. 

The structure of the directory is:
```
./sup_dataset
+-- semcor-sst.txt
+-- conll2003
|    +-- ner_train.txt
|    +-- ner_valid.txt
|    +-- ner_test.txt
+-- sst2
|    +-- sentiment-train.jsonl
|    +-- sentiment-dev.jsonl
|    +-- sentiment-test.jsonl
+-- snli
|    +-- snli_1.0_train.jsonl 
|    +-- snli_1.0_dev.jsonl 
|    +-- snli_1.0_test.jsonl 
```

To retrieve:
* Semcor-sst: Available upon request
* conll2003(NER): Download from https://www.clips.uantwerpen.be/conll2003/ner/
* wsj: Wall Street Journal (WSJ) release 3 (LDC99T42). Use sections 22-24 as test set.
* brown: Retrieve from http://www.nltk.org/nltk_data/
* sst2: Download from https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-{train,dev,test}
* snli: Retrieve from torchtext.datasets.SNLI

## Configurations

In the `configs` folder, we include a .yaml file for each downstream evaluation (Semcor, WSJ, Brown, NER, SST2, SNLI). This sets all of the hyperparameters for the word prism model, including the source embeddings, the projection layer, orthogonality constraint, final dimension, and LSTM hyperparameters. All hyperparameters are set to the final setting for testing, but feel free to play around with them (and ask us if you have any questions).

Next to `embeds_root`, write the path to the parent directory containing all of your source embeddings. Under `exp_embs`, list the source embeddings inside the parent directory that you would like to include in your word prism. Next to `dataset_dir`, write the path to the parent directory which contains the downstream evaluation data. Next to `output_dir`, write the path to the directory where you would like the results of the experiment to be outputted.

## Running Word Prisms

Now that everything is set up, you can run finally run word prisms! To run a sequence labelling task (Semcor, WSJ, Brown, NER), run the following command `python train_sequence_labelling.py -c [Path to config file corresponding to the downstream task you are running on]`. To run a classification task (SST2, SNLI), run the following command `python train_text_classification.py -c [Path to config file corresponding to the downstream task you are running on]`. 
