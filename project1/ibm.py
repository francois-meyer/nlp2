#!/usr/bin/env python

"""
Python Version: 3.6

Implementations of IBM models 1 and 2.
"""


def preprocess(line):
    """
    Apply preprocessing to line in corpus.
    :param line:
    :return:
    """

    line = line.lower() # Convert to lowercase

    return line


def get_vocab(file):
    """
    Extract all unique words from a corpus.
    :param file: text file containing corpus
    :return: set of unique words
    """

    vocab = set()
    with open(file, 'r') as f:
        for line in f:
            line = preprocess(line)
            for word in line.split():
                vocab.add(word)
    return vocab

class IBM(object):

    def __init__(self, model=1):
        self.model = model
        self.t = None

    def train(self, e_file="training/hansards.36.2.e", f_file="training/hansards.36.2.f"):

        e_vocab = get_vocab(e_file)
        f_vocab = get_vocab(f_file)

        # Model t(f|e) as t[e][f]
        self.t = self.initialise_params(e_vocab, f_vocab, model=1)


    def initialise_params(self, e_vocab, f_vocab, model=1):

        if model == 1:

            t = {e_word: {f_word: 1.0/len(f_vocab)} for f_word in f_vocab for e_word in e_vocab}

        return t

def main():
    model = IBM()
    model.train()

    print(model.t['primordial']['arrêtait'])
    print(model)

if __name__ == "__main__":
    main()