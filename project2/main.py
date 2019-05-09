
from torch import nn
from torch import optim
from utils import *
from gensim.models.word2vec import LineSentence

class RNNLM(nn.Module):

    def __init__(self, vocab, input_size, hidden_size):
        super(RNNLM, self).__init__()

        self.vocab = vocab
        self.input_dim = input_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab.size(), input_size, padding_idx=0)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.affine = nn.Linear(hidden_size, vocab.size())

        # nn.Sequential(
        #     nn.Dropout(p=0.5),  # explained later
        #     nn.Linear(hidden_dim, output_dim)
        # )

    def forward(self, input_indices):

        embeddings = self.embed(input_indices)
        outputs, hiddens = self.gru(embeddings)
        logits = self.affine(outputs)
        return logits


def train_batch(model, batch_sentences, log_softmax, criterion, optimizer):

    # Get batch and propagate forward
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits = model(input_indices)
    num_examples = target_indices.size(0) * target_indices.size(1)

    # Compute NLL loss
    log_softmax = log_softmax(logits.view([num_examples, -1]))
    loss = criterion(log_softmax, target_indices.view(-1))
    batch_loss = loss.item()

    # Backpropagate error
    model.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_loss

def train(model, sentences, epochs, batch_size, lr):

    log_softmax = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        train_loss = 0
        batch_sentences = []
        for sentence in sentences:
            batch_sentences.append(sentence)
            if len(batch_sentences) == batch_size:
                train_loss += train_batch(model, batch_sentences, log_softmax, criterion, optimizer)
                batch_sentences = []

        if len(batch_sentences) > 0:
            train_loss += train_batch(model, batch_sentences, log_softmax, criterion, optimizer)

        print("Training loss:" + str(train_loss))

def evaluate(model, sentences, batch_size):



def generate(model, n, max_len=20):

    samples = []
    with torch.no_grad():
        while len(samples) < n:
            next_word = "<SOS>"
            sample = [next_word]
            while next_word != "<EOS>" and len(sample) <= max_len:
                logits = model(torch.LongTensor([[model.vocab.word2index[next_word]]]))
                max_index = torch.argmax(logits)
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
            samples.append(sample)
    return samples


def main():

    # Training data
    train_file = "data/vw.txt" #""data/02-21.10way.clean"
    sentences = LineSentence(train_file)

    # Layer sizes
    input_size = 100
    hidden_size = 100

    # Training
    lr = 0.01
    epochs = 10
    batch_size = 10

    # Create and train model
    vocab = build_vocab(train_file)
    model = RNNLM(vocab, input_size, hidden_size)
    train(model, sentences, epochs, batch_size, lr)

    # Generate samples
    samples = generate(model, n=10)
    print(samples)

if __name__ == "__main__":
    main()