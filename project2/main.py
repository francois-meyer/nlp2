
from torch import nn
from torch import optim
from utils import *
from gensim.models.word2vec import LineSentence
import numpy as np

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


def process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=False):

    # Get batch and propagate forward
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits = model(input_indices)
    num_examples = target_indices.size(0) * target_indices.size(1)

    # Compute NLL loss
    log_softmax = to_log_softmax(logits.view([num_examples, -1]))
    loss = criterion(log_softmax, target_indices.view(-1))
    batch_loss = loss.item()

    # Backpropagate error
    if not eval:
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute word prediction accuracy
    num_correct = None
    if eval:
        predictions = logits.view([num_examples, -1]).argmax(dim=-1)
        targets = target_indices.view(-1)
        num_correct = ((predictions == targets) & (targets != model.vocab.PAD_INDEX)).sum().item()
        # print(predictions)
        # print(targets)
        # print(num_correct)

    return batch_loss, num_correct

def train(model, sentences, epochs, batch_size, lr):

    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        train_loss = 0
        batch_sentences = []
        for sentence in sentences:
            batch_sentences.append(sentence)
            if len(batch_sentences) == batch_size:
                batch_loss, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
                train_loss += batch_loss
                batch_sentences = []

        if len(batch_sentences) > 0:
            batch_loss, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
            train_loss += batch_loss

        #print("Training loss:" + str(train_loss))
        neg_loglik, perplex, acc = evaluate(model, sentences, batch_size)
        print("Validation:")
        print("NLL: " + str(neg_loglik))
        print("Perplexity: " + str(perplex))
        print("Accuracy: " + str(acc))


def evaluate(model, sentences, batch_size):

    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()
    optimizer = None

    neg_loglik = 0
    total_predictions = 0
    num_correct = 0
    batch_sentences = []
    for sentence in sentences:
        batch_sentences.append(sentence)
        if len(batch_sentences) == batch_size:
            batch_loss, batch_correct = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=True)
            neg_loglik += batch_loss
            num_correct += batch_correct
            total_predictions += sum([len(sentence)+1 for sentence in batch_sentences])
            batch_sentences = []

    if len(batch_sentences) > 0:
        batch_loss, batch_correct = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=True)
        neg_loglik += batch_loss
        num_correct += batch_correct
        total_predictions += sum([len(sentence)+1 for sentence in batch_sentences])

    perplex = np.exp(neg_loglik / total_predictions)
    acc = num_correct / total_predictions
    #
    # print(num_correct)
    # print(total_predictions)

    return neg_loglik, perplex, acc


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
    train_file = "data/mock" #""data/02-21.10way.clean"
    sentences = LineSentence(train_file)

    # Layer sizes
    input_size = 100
    hidden_size = 100

    # Training
    lr = 0.01
    epochs = 10
    batch_size = 2

    # Create and train model
    vocab = build_vocab(train_file)
    model = RNNLM(vocab, input_size, hidden_size)
    train(model, sentences, epochs, batch_size, lr)

    # Evaluate model
    neg_loglik, perplex, acc = evaluate(model, sentences, batch_size)
    print("Test:")
    print("NLL: " + str(neg_loglik))
    print("Perplexity: " + str(perplex))
    print("Accuracy: " + str(acc))

    # Generate samples
    samples = generate(model, n=10)
    print(samples)



if __name__ == "__main__":
    main()