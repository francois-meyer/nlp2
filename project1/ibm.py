#!/usr/bin/env python

"""
Python Version: 3.6

Implementations of IBM models 1 and 2.
"""

from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aer import *
import logging
logging.basicConfig(level=logging.DEBUG)


NULL_TOKEN = "<NULL>"
LIMIT = 0# how many sentences to train on

def preprocess(line):
    """
    Apply preprocessing to line in corpus.
    :param line:
    :return:
    """

    #line = line.lower()  # to lower case
    #line = re.sub(r"\d+", "", line)  # remove digits
    #line = re.sub(r'[^\w\s]', "", line)  # remove all non-alphanumeric and non-space characters
    #line = re.sub(r"\s+", " ", line).strip()  # remove excess white spaces
    return line


def get_vocab(file):
    """
    Extract all unique words from a corpus.
    :param file: text file containing corpus
    :return: set of unique words
    """

    vocab = set()
    count = 0
    with open(file, 'r') as f:
        for line in f:
            #line = preprocess(line)
            for word in line.split():
                vocab.add(word)
            count += 1
            if count == LIMIT:
                break
    return vocab

def get_corpus(e_file, f_file, add_null=True):


    fe = open(e_file)
    ff = open(f_file)
    count = 0
    for e_sent, f_sent in zip(fe, ff):

        #e_sent = preprocess(e_sent)
        if add_null:
            e_sent = NULL_TOKEN + " " + e_sent
        #f_sent = preprocess(f_sent)
        yield (e_sent.split(), f_sent.split())

        count += 1
        if count == LIMIT:
            break

class IBM(object):

    def __init__(self, model=1):
        self.model = model
        self.t = None
        self.e_vocab = None
        self.f_vocab = None
        self.valid = None
        self.plot = False
        self.test = False
        self.log_likelihoods = []
        self.valid_aers = []

        self.convergence_test_aer = None
        self.convergence_iter = None

        self.best_valid_test_aer = None
        self.best_valid_aer = 1.01
        self.best_valid_iter = None

    def train(self, e_file="training/hansards.36.2.e", f_file="training/hansards.36.2.f", iters=10,
              valid=True, plot=True, test=True):

        self.valid = valid
        self.plot = plot
        self.test = test

        logging.info("Creating English vocabulary...")
        self.e_vocab = get_vocab(e_file)
        self.e_vocab.add(NULL_TOKEN)

        logging.info("Creating French vocabulary...")
        self.f_vocab = get_vocab(f_file)

        logging.info("Initialising model parameters...")
        self.initialise_params()

        logging.info("Training parameters with EM...")
        self.EM(e_file, f_file, iters)

        if self.test:

            test_aer = None

            # Check if convergence AER has been set, if not then set it
            if self.convergence_test_aer is None:
                test_aer = self.get_aer(e_file="testing/test/test.e",
                                        f_file="testing/test/test.f",
                                        align_file="testing/answers/test.wa.nonullalign",
                                        output=True,
                                        selection="convergence")
                self.convergence_test_aer = test_aer
                self.convergence_iter = iters

            # Check if the best validation AER has been set, if not then set it
            if self.best_valid_test_aer is None:
                test_aer = self.get_aer(e_file="testing/test/test.e",
                                        f_file="testing/test/test.f",
                                        align_file="testing/answers/test.wa.nonullalign",
                                        output=True,
                                        selection="validation")
                self.best_valid_test_aer = test_aer
                self.best_valid_iter = iters

            if test_aer is None:
                test_aer = self.get_aer(e_file="testing/test/test.e",
                                        f_file="testing/test/test.f",
                                        align_file="testing/answers/test.wa.nonullalign")

            logging.info("Final test AER: " + str(test_aer))

            logging.info("Selected models:")

            logging.info("Training log likelihood converged at iteration " + str(self.convergence_iter))
            logging.info("Test AER:" + str(self.convergence_test_aer))

            logging.info("Best validation AER obtained at iteration " + str(self.best_valid_iter))
            logging.info("Test AER:" + str(self.best_valid_test_aer))

        if self.plot:
            iterations = list(range(1, len(self.log_likelihoods)+1))
            if len(self.log_likelihoods) > iters:
                iterations = [0] + iterations
                iterations = iterations[:-1]

            ax = sns.lineplot(iterations, self.log_likelihoods)
            ax.set_xlabel(xlabel="Training iterations", fontsize=14)
            ax.set_ylabel(ylabel="Training log likelihood", fontsize=14)
            plt.title("IBM Model " + str(self.model) + " - Evolution of the training log likelihood", fontsize=14)
            plt.plot(self.convergence_iter, self.log_likelihoods[self.convergence_iter -1], "r.", markersize=15)
            plt.savefig("train_ibm" + str(self.model))
            plt.show()

            ax = sns.lineplot(iterations, self.valid_aers)
            ax.set_xlabel(xlabel="Training iterations", fontsize=14)
            ax.set_ylabel(ylabel="AER on validation data", fontsize=14)
            plt.title("IBM Model " + str(self.model) + " - Evolution of the validation AER", fontsize=14)
            plt.plot(self.best_valid_iter, self.valid_aers[self.best_valid_iter - 1], "r.", markersize=15)
            plt.savefig("valid_ibm" + str(self.model))
            plt.show()


    def EM(self, e_file, f_file, iters):

        # Train parameters with EM algorithm
        for i in range(iters):

            logging.info("Starting iteration " + str(i+1))

            if self.model == 1:

                # All counts to zero for the new iteration
                pair_counts = defaultdict(float)
                word_counts = defaultdict(float)

                # Expectation step
                logging.info("Expectation step")
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
                logging.info("Maximisation step")
                for e_sent, f_sent in get_corpus(e_file, f_file):
                    for e_word in e_sent:
                        for f_word in f_sent:
                            self.t[e_word][f_word] = pair_counts[(e_word, f_word)] / word_counts[e_word]


            elif self.model == 2:
                # EM for IBM2
                pass


            logging.info("Complete")
            if self.valid:
                # Compute and store training log likelihood
                log_likelihood = self.get_log_likelihood(e_file, f_file)
                self.log_likelihoods.append(log_likelihood)
                logging.info("Training log likelihood: " + str(log_likelihood))

                # Check if training log likelihood has converged
                if self.convergence_test_aer is None and len(self.log_likelihoods) >= 2 and self.log_likelihoods[-1] < 1.001 * self.log_likelihoods[-2]:
                    test_aer = self.get_aer(e_file="testing/test/test.e",
                                            f_file="testing/test/test.f",
                                            align_file="testing/answers/test.wa.nonullalign",
                                            output=True,
                                            selection="convergence")
                    self.convergence_test_aer = test_aer
                    self.convergence_iter = i + 1

                # Compute and store validation AER
                valid_aer = self.get_aer() # on validation set
                self.valid_aers.append(valid_aer)
                logging.info("Validation AER: " + str(valid_aer))

                # Check if the current validation AER is the best so far
                if self.best_valid_aer > valid_aer:
                    self.best_valid_aer = valid_aer
                    test_aer = self.get_aer(e_file="testing/test/test.e",
                                            f_file="testing/test/test.f",
                                            align_file="testing/answers/test.wa.nonullalign",
                                            output=True,
                                            selection="validation")
                    self.best_valid_test_aer = test_aer
                    self.best_valid_iter = i + 1



    def get_aer(self, e_file="validation/dev.e", f_file="validation/dev.f", align_file="validation/dev.wa.nonullalign",
                    output=False, selection="validation"):

        gold_sets = read_naacl_alignments(align_file)

        # 2. Here you would have the predictions of your own algorithm
        predictions = []
        for e_sent, f_sent in get_corpus(e_file, f_file):

            # For each french word, find the most likely aligned english word
            links = set()
            for f_index, f_word in enumerate(f_sent):
                max_t = 0.0
                max_a = 0 # assume null aligned
                for e_index, e_word in enumerate(e_sent):
                    if self.t[e_word][f_word] > max_t:
                        max_t = self.t[e_word][f_word]
                        max_a = e_index

                if max_a != 0: # Not aligned to NULL word
                    link = (max_a, f_index+1)
                    links.add(link)

            predictions.append(links)

        # 3. Compute AER
        metric = AERSufficientStatistics()
        for gold, pred in zip(gold_sets, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        if output:
            logging.info("Writing predicted test alignments to file.")
            file_name = "ibm" + str(self.model) + ".mle.naacl_" + selection

            # Clear file
            open(file_name, 'w').close()

            # Write predictions to file
            with open(file_name, 'a') as file:
                for i, pred in enumerate(predictions):
                    for link in pred:
                        file.write(str(i+1).zfill(4) + " " + str(link[0]) + " " + str(link[1]) + " S\n")

        return metric.aer()


    def get_log_likelihood(self, e_file, f_file):

        """TODO: Maybe add normalisation for alignment probabilities"""

        log_likelihood = 0.0

        for e_sent, f_sent in get_corpus(e_file, f_file):
            sentence_likelihood = 0.0
            for f_word in f_sent:
                for e_word in e_sent:
                    sentence_likelihood += self.t[e_word][f_word]

            if sentence_likelihood > 0.0:
                log_likelihood += np.log(sentence_likelihood)

        return log_likelihood

    def initialise_params(self):

        if self.model == 1:

            # Store t(f|e) as t[e][f]
            initial_value = 1.0/len(self.f_vocab)
            #self.t = {e_word: {f_word: initial_value for f_word in self.f_vocab} for e_word in self.e_vocab}
            self.t = defaultdict(lambda: defaultdict(lambda: initial_value))

        elif self.model == 2:

            # Initialise IBM2 parameters (some of which will be the same)
            pass


def main():
    model = IBM()
    #model.train(e_file="mock/e", f_file="mock/f", iters=100)
    model.train(iters=3)

    #print(model.t['b']['x'])
    #print(model.t)

if __name__ == "__main__":
    main()