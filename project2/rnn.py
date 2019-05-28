
from torch import nn
from torch import optim
from utils import *
from gensim.models.word2vec import LineSentence
import numpy as np
import random

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

    def forward(self, input_indices, init_hidden=None):

        embeddings = self.embed(input_indices)

        if init_hidden is None:
            outputs, final_hidden = self.gru(embeddings)
        else:
            outputs, final_hidden = self.gru(embeddings, init_hidden)
        logits = self.affine(outputs)
        return logits, final_hidden


def process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=False):

    # Get batch and propagate forward
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits, final_hidden_ = model(input_indices)
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
    num_correct = num_predicted = None
    if eval:
        predictions = logits.view([num_examples, -1]).argmax(dim=-1)
        targets = target_indices.view(-1)
        num_correct = ((predictions == targets) & (targets != model.vocab.PAD_INDEX)).sum().item()
        num_predicted = (targets != model.vocab.PAD_INDEX).sum().item()

    return batch_loss, num_correct, num_predicted

def train(model, train_sentences, valid_sentences, epochs, batch_size, lr):

    print("Training model...")
    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss(ignore_index=model.vocab.PAD_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        print("Epoch " + str(i+1))
        train_loss = 0
        batch_sentences = []
        for sentence in train_sentences:
            batch_sentences.append(sentence)
            if len(batch_sentences) == batch_size:
                batch_loss, _, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
                train_loss += batch_loss
                batch_sentences = []

        if len(batch_sentences) > 0:
            batch_loss, _, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
            train_loss += batch_loss

        print("Training loss:" + str(train_loss))
        neg_loglik, perplex, acc = evaluate(model, valid_sentences, batch_size)
        print("Validation:")
        print("NLL: " + str(neg_loglik))
        print("Perplexity: " + str(perplex))
        print("Accuracy: " + str(acc))


def evaluate(model, sentences, batch_size):

    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss(ignore_index=model.vocab.PAD_INDEX, reduction="sum")
    optimizer = None

    neg_loglik = 0
    total_predictions = 0
    num_correct = 0
    num_sentences = 0
    batch_sentences = []
    for sentence in sentences:
        num_sentences += 1
        batch_sentences.append(sentence)
        if len(batch_sentences) == batch_size:
            batch_loss, batch_correct, batch_predicted = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=True)
            neg_loglik += batch_loss
            num_correct += batch_correct
            total_predictions += batch_predicted
            batch_sentences = []

    if len(batch_sentences) > 0:
        batch_loss, batch_correct, batch_predicted = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=True)
        neg_loglik += batch_loss
        num_correct += batch_correct
        total_predictions += batch_predicted
        total_predictions += sum([len(sentence)+1 for sentence in batch_sentences])

    perplex = np.exp(neg_loglik / total_predictions)
    acc = num_correct / total_predictions
    neg_loglik = neg_loglik / num_sentences

    return neg_loglik, perplex, acc


def generate(model, n, max_len=20):

    to_softmax = nn.Softmax(dim=-1)

    samples = []
    with torch.no_grad():
        while len(samples) < n:
            #next_index = random.randint(0, model.vocab.size())
            next_word = "<SOS>"
            final_hidden = None
            #_, final_hidden = model(torch.LongTensor([[model.vocab.SOS_INDEX]]), None)
            sample = ["<SOS>"]
            while next_word != "<EOS>" and len(sample) <= max_len:
                logits, final_hidden = model(torch.LongTensor([[model.vocab.word2index[next_word]]]), final_hidden)
                #max_index = torch.argmax(logits)
                softmax = to_softmax(logits.view([-1]))
                next_index = np.random.choice(model.vocab.size(), p=softmax.detach().numpy())
                next_word = model.vocab.index2word[next_index]
                sample.append(next_word)
            if next_word == "<EOS>":
                samples.append(sample)
    return samples

def generate2(model, n, max_len=20):

    to_softmax = nn.Softmax(dim=-1)

    samples = []
    with torch.no_grad():
        while len(samples) < n:
            #next_index = random.randint(0, model.vocab.size())
            next_word = "<SOS>"
            final_hidden = None
            #_, final_hidden = model(torch.LongTensor([[model.vocab.SOS_INDEX]]), None)
            sample = ["<SOS>"]
            while next_word != "<EOS>" and len(sample) <= max_len:
                logits, final_hidden = model(torch.LongTensor([[model.vocab.word2index[next_word]]]), final_hidden)

                if next_word == "<SOS>":
                    softmax = to_softmax(logits.view([-1]))
                    next_index = np.random.choice(model.vocab.size(), p=softmax.detach().numpy())
                else:
                    next_index = torch.argmax(logits)

                next_word = model.vocab.index2word[next_index]
                sample.append(next_word)
            if next_word == "<EOS>":
                samples.append(sample)
    return samples


def main():

    # Training data
    train_file = "data/train.txt" #""data/02-21.10way.clean"
    valid_file = "data/valid.txt"
    test_file = "data/test.txt"

    #train_file = valid_file = test_file = "data/vw.txt"

    train_sentences = LineSentence(train_file)
    valid_sentences = LineSentence(valid_file)
    test_sentences = LineSentence(test_file)

    # Layer sizes
    input_size = 256
    hidden_size = 256

    # Training
    lr = 0.001
    epochs = 20
    batch_size = 16

    # Create and train model
    vocab, num_sentences_ = build_vocab(train_file)
    model = RNNLM(vocab, input_size, hidden_size)
    train(model, train_sentences, valid_sentences, epochs, batch_size, lr)

    # Evaluate model
    neg_loglik, perplex, acc = evaluate(model, test_sentences, batch_size)
    print("Test:")
    print("NLL: " + str(neg_loglik))
    print("Perplexity: " + str(perplex))
    print("Accuracy: " + str(acc))

    # Generate samples
    samples = generate(model, n=100)
    print(samples)

    # Generate samples
    samples = generate2(model, n=100)
    print(samples)


if __name__ == "__main__":
    main()