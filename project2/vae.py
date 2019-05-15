
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils import *
from gensim.models.word2vec import LineSentence
import numpy as np

from scipy.stats import multivariate_normal

class SentenceVAE(nn.Module):

    def __init__(self, vocab, input_size, hidden_size, latent_size):
        super(SentenceVAE, self).__init__()

        self.vocab = vocab
        self.input_dim = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder
        self.embed = nn.Embedding(vocab.size(), input_size, padding_idx=0)
        self.gru_encode = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        #self.affine_encoder = nn.Linear(hidden_size*2, vocab.size())

        self.dense_mean = nn.Linear(hidden_size*2, self.latent_size)
        self.dense_sd = nn.Linear(hidden_size*2, self.latent_size)
        self.softplus = nn.Softplus()

        # Decoder
        self.affine_init = nn.Linear(latent_size, hidden_size)
        self.tanh = nn.Tanh()
        self.gru_decoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.affine_decoder = nn.Linear(hidden_size, vocab.size())

    def forward(self, input_indices, predict=False):

        batch_size = input_indices.size(0)

        # Encoder
        embeddings = self.embed(input_indices)
        outputs, final_hidden = self.gru_encode(embeddings)
        #h = self.affine_encoder(final_hidden)
        h = final_hidden.view(batch_size, self.hidden_size * 2)

        mean = self.dense_mean(h)
        sd = self.softplus(self.dense_sd(h))

        if not predict:
            epsilon = Variable(torch.randn([batch_size, self.latent_size]))
            z = mean + epsilon * sd
        else:
            z = mean

        # Decoder
        init = self.tanh(self.affine_init(z))
        outputs, final_hidden = self.gru_decoder(embeddings, init.unsqueeze(0))
        logits = self.affine_decoder(outputs)

        return logits, mean, sd

    def encode(self, input_indices):

        batch_size = input_indices.size(0)

        # Encoder
        embeddings = self.embed(input_indices)
        outputs, final_hidden = self.gru_encode(embeddings)
        #h = self.affine_encoder(final_hidden)
        h = final_hidden.view(batch_size, self.hidden_size * 2)

        mean = self.dense_mean(h)
        sd = self.softplus(self.dense_sd(h))
        epsilon = Variable(torch.randn([batch_size, self.latent_size]))
        z = mean + epsilon * sd

        return mean, sd

    def decode(self, input_indices, init_hidden, latent=True):

        embeddings = self.embed(input_indices)

        # Decoder
        if latent:
            init_hidden = self.tanh(self.affine_init(init_hidden)).unsqueeze(0)
        outputs, final_hidden = self.gru_decoder(embeddings, init_hidden)
        logits = self.affine_decoder(outputs)

        return logits, final_hidden


def train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer):

    # Get batch and propagate forward
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits, means, sds = model(input_indices)
    num_examples = target_indices.size(0) * target_indices.size(1)

    # Compute NLL loss
    log_softmax = to_log_softmax(logits.view([num_examples, -1]))
    nll_loss = criterion(log_softmax, target_indices.view(-1))
    #nll_loss = nll_loss.item()

    # Compute KL divergece
    kld = 0.5 * torch.sum(-torch.log(sds.pow(2)) - 1 + means.pow(2) + sds.pow(2))

    # Compute ELBO
    elbo = nll_loss + kld

    model.zero_grad()
    nll_loss.backward()
    optimizer.step()

    return elbo


def eval_batch(model, batch_sentences, num_samples=10):

    # Compute word predictions
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits, means, sds = model(input_indices, predict=True)
    num_examples = target_indices.size(0) * target_indices.size(1)
    batch_size = len(batch_sentences)

    predictions = logits.view([num_examples, -1]).argmax(dim=-1)
    targets = target_indices.view(-1)
    num_correct = ((predictions == targets) & (targets != model.vocab.PAD_INDEX)).sum().item()

    # Approximate NLL with importance sampling
    total = 0
    for i in range(batch_size):
        # Sample importance distribution q
        epsilon = Variable(torch.randn([num_samples, model.latent_size]))
        samples = means[i] + epsilon * sds[i]

        # Evaluate importance pdf q(z|x)
        qs = multivariate_normal.pdf(samples.detach().numpy(), mean=means[i].detach().numpy(), cov=sds[i].detach().numpy())

        # Evaluate nominal pdf p(z, x)
        to_softmax = nn.Softmax(dim=-1)
        sentence_input_indices = input_indices[i].repeat([num_samples, 1])
        logits, final_hidden_ = model.decode(sentence_input_indices, samples)
        softmax = to_softmax(logits)
        sentence_target_indices = target_indices[i].repeat([num_samples, 1]).view(num_samples, target_indices[i].size(0), 1)
        target_softmax = torch.gather(softmax, -1, sentence_target_indices).view(num_samples, target_indices[i].size(0))
        ps = torch.prod(target_softmax, -1)

        # Average importance samples
        sample_values = ps.detach().numpy() / qs
        sample_mean = sample_values.mean()
        total += np.log(sample_mean)

    nll_estimate = -total / batch_size
    return nll_estimate, num_correct


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
                batch_loss = train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
                train_loss += batch_loss
                batch_sentences = []

        if len(batch_sentences) > 0:
            batch_loss = train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
            train_loss += batch_loss

        print("ELBO:" + str(train_loss))
        neg_loglik, perplex, acc = evaluate(model, valid_sentences, batch_size)
        print("Validation:")
        print("NLL: " + str(neg_loglik))
        print("Perplexity: " + str(perplex))
        print("Accuracy: " + str(acc))


def evaluate(model, sentences, batch_size):

    neg_loglik = 0
    total_predictions = 0
    num_correct = 0
    batch_sentences = []
    for sentence in sentences:
        batch_sentences.append(sentence)
        if len(batch_sentences) == batch_size:
            batch_loss, batch_correct = eval_batch(model, batch_sentences)
            neg_loglik += batch_loss
            num_correct += batch_correct
            total_predictions += sum([len(sentence)+1 for sentence in batch_sentences])
            batch_sentences = []

    if len(batch_sentences) > 0:
        batch_loss, batch_correct = eval_batch(model, batch_sentences)
        neg_loglik += batch_loss
        num_correct += batch_correct
        total_predictions += sum([len(sentence)+1 for sentence in batch_sentences])

    perplex = np.exp(neg_loglik / total_predictions)
    acc = num_correct / total_predictions


    return neg_loglik, perplex, acc


def generate(model, n, max_len=20):

    samples = []
    with torch.no_grad():
        while len(samples) < n:
            next_word = "<SOS>"
            sample = [next_word]
            init_hidden = Variable(torch.randn(1, model.latent_size))
            latent = True
            while next_word != "<EOS>" and len(sample) <= max_len:
                logits, init_hidden = model.decode(torch.LongTensor([[model.vocab.word2index[next_word]]]), init_hidden, latent)
                latent = False
                max_index = torch.argmax(logits)
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
            samples.append(sample)
    return samples

def reconstruct_sentence(model, sentence, num_samples=10, max_len=20):

    # Get posterior distribution
    input_indices, target_indices = get_batch([sentence], model.vocab)
    mean, sd = model.encode(input_indices)

    # Generate samples from this
    samples = []
    with torch.no_grad():
        # Use the approximate posterior mean
        next_word = "<SOS>"
        sample = [next_word]
        init_hidden = Variable(mean)
        latent = True
        while next_word != "<EOS>" and len(sample) <= max_len:
            logits, init_hidden = model.decode(torch.LongTensor([[model.vocab.word2index[next_word]]]), init_hidden, latent)
            latent = False
            max_index = torch.argmax(logits)
            next_word = model.vocab.index2word[max_index]
            sample.append(next_word)
        samples.append(sample)

        # Use 10 samples from ~z
        while len(samples) < num_samples+1:
            next_word = "<SOS>"
            sample = [next_word]
            epsilon = Variable(torch.randn(1, model.latent_size))
            init_hidden = mean + epsilon * sd
            latent = True
            while next_word != "<EOS>" and len(sample) <= max_len:
                logits, init_hidden = model.decode(torch.LongTensor([[model.vocab.word2index[next_word]]]), init_hidden, latent)
                latent = False
                max_index = torch.argmax(logits)
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
            samples.append(sample)

    return samples

def main():

    # Training data
    train_file = "data/train.txt" #""data/02-21.10way.clean"
    train_sentences = LineSentence(train_file)
    valid_file = "data/valid.txt"
    valid_sentences = LineSentence(valid_file)
    test_file = "data/test.txt"
    test_sentences = LineSentence(test_file)

    # Layer sizes
    input_size = 100
    hidden_size = 100
    latent_size = 16

    # Training
    lr = 0.001
    epochs = 10
    batch_size = 32

    # Create and train model
    vocab = build_vocab(train_file)
    print(vocab)
    model = SentenceVAE(vocab, input_size, hidden_size, latent_size)
    train(model, train_sentences, valid_sentences, epochs, batch_size, lr)

    # Evaluate model
    neg_loglik, perplex, acc = evaluate(model, test_sentences, batch_size)
    print("Test:")
    print("NLL: " + str(neg_loglik))
    print("Perplexity: " + str(perplex))
    print("Accuracy: " + str(acc))

    # Generate samples
    samples = generate(model, n=10)
    print(samples)

    # Reconstruct sentence
    sentence = "<SOS> Anger wants a voice <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)


if __name__ == "__main__":
    main()