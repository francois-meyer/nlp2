# Here we first define a class that can map a word to an ID (w2i)
# and back (i2w).

from collections import Counter
import torch

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


def build_vocab(corpus_file):

    # Go through each sentence in the corpus, adding words to the vocabulary
    vocab = Vocab()
    vocab.update("<PAD>")
    vocab.update("<UNK>")
    vocab.update("<SOS>")
    vocab.update("<EOS>")

    with open(corpus_file) as file:
        for line in file:
            for token in line.split():
                vocab.update(token)

    return vocab

def get_batch(sentences, vocab, batch_size):

    input_indices = []
    target_indices = []

    count = 0

    for sentence in sentences:
        sentence_indices = [vocab.get_index(w) if vocab.contains(w) else vocab.UNK_INDEX for w in sentence]
        input_indices.append([vocab.SOS_INDEX] + sentence_indices)
        target_indices.append(sentence_indices + [vocab.EOS_INDEX])

        count += 1
        if count == batch_size:
            break


    max_length = max([len(sent) for sent in input_indices])
    input_indices = [sentence_indices + [vocab.PAD_INDEX] * (max_length - len(sentence_indices)) for sentence_indices in input_indices]
    target_indices = [sentence_indices + [vocab.PAD_INDEX] * (max_length - len(sentence_indices)) for sentence_indices in target_indices]
    return torch.LongTensor(input_indices), torch.LongTensor(target_indices)