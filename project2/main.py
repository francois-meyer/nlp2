
from torch import nn
from torch import optim
from utils import *
from gensim.models.word2vec import LineSentence

import logging
logging.basicConfig(level=logging.DEBUG)


class RNNLM(nn.Module):

    def __init__(self, vocab, input_size, hidden_size):
        super(RNNLM, self).__init__()

        self.vocab = vocab
        self.input_dim = input_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab.size(), input_size, padding_idx=0)
        self.gru = nn.GRU(input_size, hidden_size)
        self.affine = nn.Linear(hidden_size, vocab.size())

        # nn.Sequential(
        #     nn.Dropout(p=0.5),  # explained later
        #     nn.Linear(hidden_dim, output_dim)
        # )

    def forward(self, input_indices):

        embeddings = self.embed(input_indices)
        outputs, hiddens = self.gru(embeddings)
        logits = self.affine(hiddens)
        return logits


def train(model, vocab, sentences, epochs, batch_size):

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs):

        train_loss = 0

        next_batch = True
        while next_batch:

            input_indices, target_indices = get_batch(sentences, vocab, batch_size)
            logits = model(input_indices)
            #targets = target_indices.vi

            loss = cross_entropy_loss(logits, target_indices)
            train_loss += loss.item()

            # erase previous gradients
            model.zero_grad()

            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

        logging.info("Training loss:")

def main():

    # Training data
    train_file = "data/02-21.10way.clean"
    sentences = LineSentence(train_file)

    # Layer sizes
    input_size = 100
    hidden_size = 100

    # Training iterations
    epochs = 10
    batch_size = 10

    # Create and train model
    vocab = build_vocab(train_file)
    model = RNNLM(vocab, input_size, hidden_size)
    train(model, vocab, sentences, epochs, batch_size)


if __name__ == "__main__":
    main()