#!/usr/bin/env python

"""
Python Version: 3.6

Implementations of IBM models 1 and 2.
"""

from collections import defaultdict

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

def get_corpus(e_file, f_file):

    fe = open(e_file)
    ff = open(f_file)
    for e_sent, f_sent in zip(fe, ff):
        yield e_sent.split(), f_sent.split()


class IBM(object):

    def __init__(self, model=1):
        self.model = model
        self.t = None
        self.e_vocab = None
        self.f_vocab = None

    def train(self, e_file="training/hansards.36.2.e", f_file="training/hansards.36.2.f", iters=10):

        self.e_vocab = get_vocab(e_file)
        self.f_vocab = get_vocab(f_file)

        self.initialise_params()
        self.EM(e_file, f_file, iters)


    def EM(self, e_file, f_file, iters):

        # Train parameters with EM algorithm
        for i in range(iters):


            if self.model == 1:

                # All counts to zero for the new iteration
                pair_counts = defaultdict(float)
                word_counts = defaultdict(float)

                # Expectation step
                for e_sent, f_sent in get_corpus(e_file, f_file):

                    normalise = {}
                    for f_word in f_sent:

                        # Sum translation probabilities of f words over all e words
                        normalise[f_word] = 0.0
                        for e_word in e_sent:
                            normalise[f_word] += self.t[e_word][f_word]

                        # Update counts
                        for e_word in e_sent:
                            delta = self.t[e_word][f_word] / normalise[f_word]
                            pair_counts[(e_word, f_word)] += delta
                            word_counts[e_word] += delta

                # Maximisation step
                for e_word in self.e_vocab:
                    for f_word in self.f_vocab:
                        self.t[e_word][f_word] = pair_counts[(e_word, f_word)] / word_counts[e_word]


            elif self.model == 2:
                # EM for IBM2
                pass


    def initialise_params(self):

        if self.model == 1:

            # Store t(f|e) as t[e][f]
            initial_value = 1.0/len(self.f_vocab)
            self.t = {e_word: {f_word: initial_value for f_word in self.f_vocab} for e_word in self.e_vocab}

        elif self.model == 2:

            # Initialise IBM2 parameters (some of which will be the same)
            pass


def main():
    model = IBM()
    model.train(e_file="mock/e", f_file="mock/f")
    #model.train()

    #print(model.t['b']['x'])
    print(model.t)

if __name__ == "__main__":
    main()