# Here we first define a class that can map a word to an ID (w2i)
# and back (i2w).

from collections import Counter
import torch
import re

class Vocab(object):

    def __init__(self):
        self.counter = Counter()
        self.word2index = {}
        self.index2word = []

        self.PAD_INDEX = 0
        self.UNK_INDEX = 1
        self.SOS_INDEX = 2
        self.EOS_INDEX = 3


    def update(self, word):
        self.counter.update([word])
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word.append(word)

    def contains(self, word):
        return word in self.word2index

    def get_index(self, word):
        return self.word2index[word]

    def size(self):
        return len(self.word2index)

    def finalise(self, min_count):

        # Discard words that occur less than min_count
        print("Raw vocab size is " + str(self.size()) + ".")
        discard_count = 0
        raw_train_count = 0
        train_count = 0
        for word in list(self.counter):
            raw_train_count += self.counter[word]
            if self.counter[word] < min_count and self.word2index[word] > 3:
                del self.word2index[word]
                self.index2word.remove(word)
                del self.counter[word]
                discard_count += 1
            else:
                train_count += self.counter[word]

        for index, word in enumerate(self.index2word):
            self.word2index[word] = index

        print("Discarded " + str(discard_count) + " words with count less than " + str(min_count) + ".")
        print("Final vocab size is " + str(self.size()) + ".")
        print("Training words down from " + str(raw_train_count) + " to " + str(train_count) + ".")


def build_vocab(corpus_file):

    print("Building vocabulary...")

    # Go through each sentence in the corpus, adding words to the vocabulary
    vocab = Vocab()
    vocab.update("<PAD>")
    vocab.update("<UNK>")
    vocab.update("<SOS>")
    vocab.update("<EOS>")

    num_sentences = 0
    with open(corpus_file) as file:
        for line in file:
            num_sentences += 1
            for token in line.split():
                vocab.update(token)
    vocab.finalise(min_count=2)
    print("Created vocabulary of size: " + str(vocab.size()))

    # count = 0
    # for word in vocab.word2index:
    #     if vocab.counter[word] < 3:
    #         vocab.word2index[word] = vocab.UNK_INDEX
    #         count += 1
    # print("Discarded " + str(count) + " words.")

    return vocab, num_sentences


def get_batch(sentences, vocab):

    input_indices = []
    target_indices = []

    for sentence in sentences:
        sentence_indices = [vocab.get_index(w) if vocab.contains(w) else vocab.UNK_INDEX for w in sentence]
        input_indices.append([vocab.SOS_INDEX] + sentence_indices)
        target_indices.append(sentence_indices + [vocab.EOS_INDEX])

    max_length = max([len(sent) for sent in input_indices])
    input_indices = [sentence_indices + [vocab.PAD_INDEX] * (max_length - len(sentence_indices)) for sentence_indices in input_indices]
    target_indices = [sentence_indices + [vocab.PAD_INDEX] * (max_length - len(sentence_indices)) for sentence_indices in target_indices]
    return torch.LongTensor(input_indices), torch.LongTensor(target_indices)