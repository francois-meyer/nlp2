
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

    def forward(self, input_indices):

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

        # Decoder
        init = self.tanh(self.affine_init(z))
        outputs, final_hidden = self.gru_decoder(embeddings, init.unsqueeze(0))
        logits = self.affine_decoder(outputs)

        return logits, mean, sd

    def decode(self, input_indices, z):

        embeddings = self.embed(input_indices)

        # Decoder
        init = self.tanh(self.affine_init(z))
        outputs, final_hidden = self.gru_decoder(embeddings, init.unsqueeze(0))
        logits = self.affine_decoder(outputs)

        return logits


def process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, eval=False, num_samples=10):

    # Get batch and propagate forward
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits, means, sds = model(input_indices)
    num_examples = target_indices.size(0) * target_indices.size(1)
    batch_size = len(batch_sentences)

    # Compute NLL loss
    log_softmax = to_log_softmax(logits.view([num_examples, -1]))
    nll_loss = criterion(log_softmax, target_indices.view(-1))
    nll_loss = nll_loss.item()

    # Compute KL divergece
    kld = 0.5 * torch.sum(-torch.log(sds.pow(2)) - 1 + means.pow(2) + sds.pow(2))

    # Compute ELBO
    elbo = nll_loss + kld

    # Backpropagate error
    if not eval:
        model.zero_grad()
        elbo.backward()
        optimizer.step()

    # Compute word prediction accuracy
    num_correct = None
    if eval:
        # Compute word predictions
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
            logits = model.decode(sentence_input_indices, samples)
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

    return elbo, num_correct

def train(model, train_sentences, valid_sentences, epochs, batch_size, lr):

    print("Training model...")
    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        print("Epoch " + str(i+1))
        train_loss = 0
        batch_sentences = []
        for sentence in train_sentences:
            batch_sentences.append(sentence)
            if len(batch_sentences) == batch_size:
                batch_loss, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
                train_loss += batch_loss
                batch_sentences = []

        if len(batch_sentences) > 0:
            batch_loss, _ = process_batch(model, batch_sentences, to_log_softmax, criterion, optimizer)
            train_loss += batch_loss

        print("ELBO:" + str(train_loss))
        neg_loglik, perplex, acc = evaluate(model, valid_sentences, batch_size)
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
    train_file = "data/vw.txt" #""data/02-21.10way.clean"
    train_sentences = LineSentence(train_file)
    valid_file = "data/vw.txt"
    valid_sentences = LineSentence(valid_file)
    test_file = "data/vw.txt"
    test_sentences = LineSentence(test_file)

    # Layer sizes
    input_size = 100
    hidden_size = 100
    latent_size = 16

    # Training
    lr = 0.01
    epochs = 10
    batch_size = 10

    # Create and train model
    vocab = build_vocab(train_file)
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



if __name__ == "__main__":
    main()