
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils import *
from gensim.models.word2vec import LineSentence
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
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
        self.gru_encode = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.affine_encoder = nn.Linear(hidden_size, latent_size*2)

        self.dense_mean = nn.Linear(latent_size*2, self.latent_size)
        self.dense_sd = nn.Linear(latent_size*2, self.latent_size)
        self.softplus = nn.Softplus()

        # Decoder
        self.affine_init = nn.Linear(latent_size, hidden_size)
        self.tanh = nn.Tanh()
        self.gru_decoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.affine_decoder = nn.Linear(hidden_size, vocab.size())

    def forward(self, input_indices, predict=False, eval=False):

        batch_size = input_indices.size(0)

        # Encoder
        embeddings = self.embed(input_indices)
        outputs, final_hidden = self.gru_encode(embeddings)
        h = final_hidden.view(batch_size, self.hidden_size)
        h = self.affine_encoder(h)

        mean = self.dense_mean(h)
        sd = self.softplus(self.dense_sd(h))

        if not predict:
            epsilon = Variable(torch.randn([batch_size, self.latent_size]))
            z = mean + epsilon * sd
        else:
            z = mean

        # Decoder
        if not eval:
            prob = torch.rand(input_indices.size())
            prob[(input_indices.data - self.vocab.SOS_INDEX) * (input_indices.data - self.vocab.PAD_INDEX) == 0] = 1
            decoder_input_sequence = input_indices.clone()
            decoder_input_sequence[prob < 0.75] = self.vocab.UNK_INDEX
            embeddings = self.embed(decoder_input_sequence)

        init = self.tanh(self.affine_init(z))
        outputs, final_hidden = self.gru_decoder(embeddings, init.unsqueeze(0))
        logits = self.affine_decoder(outputs)

        return logits, mean, sd

    def encode(self, input_indices):

        batch_size = input_indices.size(0)

        # Encoder
        embeddings = self.embed(input_indices)
        outputs, final_hidden = self.gru_encode(embeddings)
        h = final_hidden.view(batch_size, self.hidden_size)
        h = self.affine_encoder(h)

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


def train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, kl_weight):

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
    elbo = nll_loss + kl_weight * torch.max(torch.FloatTensor([5.0*len(batch_sentences)]), kld)

    # print("---------------------------")
    # print(5.0*len(batch_sentences))
    # print(nll_loss)
    # print(torch.max(torch.FloatTensor([5.0*len(batch_sentences)]), nll_loss))

    model.zero_grad()
    elbo.backward()
    optimizer.step()

    return elbo, kld


def eval_batch_is(model, batch_sentences, num_samples=20):

    # Compute word predictions
    input_indices, target_indices = get_batch(batch_sentences, model.vocab)
    logits, means, sds = model(input_indices, predict=True, eval=True)
    num_examples = target_indices.size(0) * target_indices.size(1)
    batch_size = len(batch_sentences)
    batch_kl = 0.5 * torch.sum(-torch.log(sds.pow(2)) - 1 + means.pow(2) + sds.pow(2))

    predictions = logits.view([num_examples, -1]).argmax(dim=-1)
    targets = target_indices.view(-1)
    num_correct = ((predictions == targets) & (targets != model.vocab.PAD_INDEX)).sum().item()
    num_predicted = (targets != model.vocab.PAD_INDEX).sum().item()

    # Approximate NLL with importance sampling
    to_log_softmax = nn.LogSoftmax(dim=-1)
    log_total = 0
    for i in range(batch_size):
        # Sample importance distribution q
        epsilon = Variable(torch.randn([num_samples, model.latent_size]))
        samples = means[i] + epsilon * sds[i]

        # Evaluate importance pdf log q(z|x)
        log_qzx = torch.Tensor(multivariate_normal.logpdf(samples.detach().numpy(), mean=means[i].detach().numpy(), cov=sds[i].detach().numpy()))

        # Evaluate prior pdf log p(z)
        log_pz = torch.Tensor(multivariate_normal.logpdf(samples.detach().numpy(), mean=np.zeros(model.latent_size),
                                                          cov=np.ones(model.latent_size)))

        # Ignore PAD tokens
        EOS_index = np.where(target_indices[i].detach().numpy() == model.vocab.EOS_INDEX)[0][0]
        length = len(input_indices[i]) - EOS_index + 1

        # Evaluate nominal pdf log p(z, x) = log p(x|z) + log p(z)r
        sentence_input_indices = input_indices[i][0:length].repeat([num_samples, 1])
        logits, final_hidden_ = model.decode(sentence_input_indices, samples)
        log_softmax = to_log_softmax(logits)
        #print(log_softmax.detach().numpy())
        sentence_target_indices = target_indices[i][0:length].repeat([num_samples, 1]).view(num_samples, target_indices[i][0:length].size(0), 1)
        target_log_softmax = torch.gather(log_softmax, -1, sentence_target_indices).view(num_samples, target_indices[i][0:length].size(0))
        log_pxz = torch.sum(target_log_softmax, -1)

        # Average importance samples
        sample_values = log_pz + log_pxz - log_qzx
        sample_mean = torch.logsumexp(sample_values, dim=0) - torch.log(torch.Tensor([num_samples]))
        log_total += sample_mean.detach().numpy()[0]

    nll_estimate = -log_total
    return nll_estimate, num_correct, num_predicted, batch_kl


def eval_batch(model, batch_sentences, criterion):
    with torch.no_grad():
        to_log_softmax = nn.LogSoftmax(dim=-1)
        # Get batch and propagate forward
        input_indices, target_indices = get_batch(batch_sentences, model.vocab)
        logits, means, sds = model(input_indices, eval=True)
        num_examples = target_indices.size(0) * target_indices.size(1)

        # Compute NLL loss
        log_softmax = to_log_softmax(logits.view([num_examples, -1]))
        loss = criterion(log_softmax, target_indices.view(-1))
        batch_loss = loss.item()
        batch_kl = 0.5 * torch.sum(-torch.log(sds.pow(2)) - 1 + means.pow(2) + sds.pow(2))

        # Compute word prediction accuracy
        num_correct = num_predicted = None
        predictions = logits.view([num_examples, -1]).argmax(dim=-1)
        targets = target_indices.view(-1)
        num_correct = ((predictions == targets) & (targets != model.vocab.PAD_INDEX)).sum().item()
        num_predicted = (targets != model.vocab.PAD_INDEX).sum().item()

        return batch_loss, num_correct, num_predicted, batch_kl

def evaluate(model, sentences, batch_size):
    criterion = nn.NLLLoss(ignore_index=model.vocab.PAD_INDEX, reduction="sum")
    neg_loglik = 0
    total_kl = 0
    total_predictions = 0
    num_correct = 0
    num_sentences = 0
    batch_sentences = []
    for sentence in sentences:
        num_sentences += 1
        batch_sentences.append(sentence)
        if len(batch_sentences) == batch_size:
            batch_loss, batch_correct, batch_predicted, batch_kl = eval_batch(model, batch_sentences, criterion)
            neg_loglik += batch_loss
            num_correct += batch_correct
            total_predictions += batch_predicted
            total_kl += batch_kl
            batch_sentences = []

    if len(batch_sentences) > 0:
        batch_loss, batch_correct, batch_predicted, batch_kl = eval_batch(model, batch_sentences, criterion)
        neg_loglik += batch_loss
        num_correct += batch_correct
        total_predictions += batch_predicted
        total_kl += batch_kl

    perplex = np.exp(neg_loglik / total_predictions)
    acc = num_correct / total_predictions
    neg_loglik = neg_loglik / num_sentences

    return neg_loglik, perplex, acc, total_kl


def train(model, train_sentences, valid_sentences, epochs, batch_size, lr, num_sentences):

    print("Training model...")
    to_log_softmax = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss(ignore_index=model.vocab.PAD_INDEX, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        epoch_step = 1/epochs
        epoch_start = i/epochs
        epoch_end = (i+1)/epochs

        print("---------------------------------------------------------------------")
        print("Epoch " + str(i+1))
        train_loss = 0
        train_kl = 0
        batch_sentences = []
        count_sentences = 0.0
        for sentence in train_sentences:
            count_sentences += 1
            batch_sentences.append(sentence)
            if len(batch_sentences) == batch_size:
                kl_weight = 1.0 # epoch_start + (count_sentences/num_sentences) * epoch_step
                batch_loss, batch_kl = train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, kl_weight)
                train_loss += batch_loss
                train_kl += batch_kl
                batch_sentences = []

        if len(batch_sentences) > 0:
            kl_weight = 1.0 # epoch_end
            batch_loss, batch_kl = train_batch(model, batch_sentences, to_log_softmax, criterion, optimizer, kl_weight)
            train_loss += batch_loss
            train_kl += batch_kl

        print("Training")
        print("ELBO:" + str(train_loss))
        print("KL:" + str(train_kl))
        print("---------------------------------------------------------------------")
        neg_loglik, perplex, acc, total_kl = evaluate(model, valid_sentences, batch_size)
        print("Validation:")
        print("NLL: " + str(neg_loglik))
        print("Perplexity: " + str(perplex))
        print("Accuracy: " + str(acc))
        print("KL: " + str(total_kl))
        print("---------------------------------------------------------------------")



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
                max_index = torch.argmax(logits[:,:,2:]) + 2
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
            samples.append(sample)
    return samples


def reconstruct_sentence(model, sentence, num_samples=20, max_len=20):

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
            max_index = torch.argmax(logits[:,:,2:]) + 2
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
                max_index = torch.argmax(logits[:,:,2:]) + 2
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
            samples.append(sample)

    return samples

def homotopy(model, sentence1, sentence2, num_samples=20):

    # Get latent represenations
    input_indices, target_indices_ = get_batch([sentence1], model.vocab)
    z1, sd_ = model.encode(input_indices)

    input_indices, target_indices_ = get_batch([sentence2], model.vocab)
    z2, sd_ = model.encode(input_indices)

    # Compute linear interpolations
    step_size = 1.0/num_samples
    alphas = list(np.arange(0, 1.01, step_size))

    samples = []
    with torch.no_grad():
        for alpha in alphas:
            next_word = "<SOS>"
            sample = [next_word]
            interpolation = alpha * z2 + (1 - alpha) * z1
            init_hidden = Variable(interpolation)
            latent = True
            while next_word != "<EOS>":
                logits, init_hidden = model.decode(torch.LongTensor([[model.vocab.word2index[next_word]]]), init_hidden,
                                                   latent)
                latent = False
                max_index = torch.argmax(logits[:,:,2:]) + 2
                next_word = model.vocab.index2word[max_index]
                sample.append(next_word)
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
    latent_size = 16

    # Training
    lr = 0.001
    epochs = 20
    batch_size = 16

    # Create and train model
    vocab, num_sentences = build_vocab(train_file)
    model = SentenceVAE(vocab, input_size, hidden_size, latent_size)
    train(model, train_sentences, valid_sentences, epochs, batch_size, lr, num_sentences)

    # Evaluate model
    neg_loglik, perplex, acc, total_kl = evaluate(model, test_sentences, batch_size)
    print("Test:")
    print("NLL: " + str(neg_loglik))
    print("Perplexity: " + str(perplex))
    print("Accuracy: " + str(acc))
    print("KL: " + str(total_kl))

    # Generate samples
    samples = generate(model, n=10)
    print(samples)

    sentence = "<SOS> the company s u s performance was helped by a record quarter for new customers he said <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    sentence = "<SOS> debt burdens are heavier <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    sentence = "<SOS> the projects are big <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    # Reconstruct sentence
    sentence = "<SOS> when the little guy gets frightened the big guys hurt badly <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    # Reconstruct sentence
    sentence = "<SOS> in that case there will be plenty of blame to go around <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    # Reconstruct sentence
    sentence = "<SOS> a dow spokeswoman declined to comment on the estimates <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    # Reconstruct sentence
    sentence = "<SOS> the financial services industry was battered by the crash <EOS>"
    samples = reconstruct_sentence(model, sentence)
    print(samples)

    # Sample homotopies between two sentences
    sentence1 = "<SOS> growth is slower <EOS>"
    sentence2 = "<SOS> profits are softer <EOS>"
    samples = homotopy(model, sentence1, sentence2, num_samples=20)
    print(samples)

    # Sample homotopies between two sentences
    sentence1 = "<SOS> a lot of pent up demand is gone <EOS>"
    sentence2 = "<SOS> that was offset by strength elsewhere <EOS>"
    samples = homotopy(model, sentence1, sentence2, num_samples=20)
    print(samples)

    # Sample homotopies between two sentences
    sentence1 = "<SOS> a few hours later the stock market dropped points <EOS>"
    sentence2 = "<SOS> analysts do n t see it that way <EOS>"
    samples = homotopy(model, sentence1, sentence2, num_samples=20)
    print(samples)

    # Sample homotopies between two sentences
    sentence1 = "<SOS> that was offset by strength elsewhere <EOS>"
    sentence2 = "<SOS> they learned they could survive it without much problem <EOS>"
    samples = homotopy(model, sentence1, sentence2, num_samples=20)
    print(samples)

    # Sample homotopies between two sentences
    sentence1 = "<SOS> ibm closed at down <EOS>"
    sentence2 = "<SOS> that did not happen<EOS>"
    samples = homotopy(model, sentence1, sentence2, num_samples=20)
    print(samples)


if __name__ == "__main__":
    main()
