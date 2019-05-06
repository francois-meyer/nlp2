
from torch import nn
from utils import *
from gensim.models.word2vec import LineSentence


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

    def forward(self, input_indices, target_indices):

        embeddings = self.embed(input_indices)
        outputs, hiddens = self.gru(embeddings)
        logits = self.affine(hiddens)
        return logits


def train(model, sentences, epochs, bach_size):

    model = RNNLM()
    cross_entropy_loss = nn.CrossEntropyLoss()

    for i in range(epochs):

        train_loss = 0

        next_batch = True
        while next_batch

            input_indices, target_indices = get_batch(sentences)
            logits = model(input_indices)
            targets = target_indices.vi

            loss = cross_entropy_loss(logits, targets)
            batchitem()

            # backward pass
            # Tip: check the Introduction to PyTorch notebook.

            # erase previous gradients
            model.zero_grad()

            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()


def main():

    train_file = "data/02-21.10way.clean"
    sentences = LineSentence(train_file)
    epochs = 10
    batch_size = 100

    model = RNNLM()
    model.train(model, sentences, epochs, batch_size)

if __name__ == "__main__":
    main()