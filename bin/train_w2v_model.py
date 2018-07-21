import click
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.onnx
from gensim.models import KeyedVectors

from tensorboardX import SummaryWriter

from word2vec_lstm.data import Corpus
from word2vec_lstm.model import RNNModel

@click.command()
@click.option('--data_path', type=str,
                    help='location of the data corpus')
@click.option('--model_type', type=str, default='LSTM',
                    help='type of recurrent net '
                         '(RNN_TANH, RNN_RELU, LSTM, GRU)')
@click.option('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
@click.option('--nlayers', type=int, default=2,
                    help='number of layers')
@click.option('--lr', type=float, default=0.001,
                    help='initial adam learning rate')
@click.option('--clip', type=float, default=0.25,
                    help='gradient clipping')
@click.option('--epochs', type=int, default=400,
                    help='upper epoch limit')
@click.option('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
@click.option('--bptt', type=int, default=600,
                    help='sequence length')
@click.option('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
@click.option('--tied', default=True,
                    help='tie the word embedding and softmax weights')
@click.option('--seed', type=int, default=1111,
                    help='random seed')
@click.option('--log-interval', type=int, default=50, help='report interval')
def run(data_path: str,
        model_type: str,
        nhid: int,
        nlayers: int,
        lr: float,
        clip: float,
        epochs: int,
        batch_size: int,
        bptt: int,
        dropout: float,
        tied: bool,
        seed: int,
        log_interval: int):
    print(f"nhid: {nhid}")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')

    ###########################################################################
    # Load data
    ###########################################################################

    corpus = Corpus(data_path)

    print(f'corpus size: {len(corpus.dictionary)}')
    emb_dim = 300

    # Create embedding matrix from corpus
    w2v_path = 'data/GoogleNews-vectors-negative300.bin'
    print(f'reading word2vec trained model from {w2v_path}')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    matrix_len = len(corpus.dictionary)
    weights_matrix = torch.empty((matrix_len, emb_dim))
    weights_matrix_labels = list()
    words_found = 0

    for i, word in enumerate(corpus.dictionary.word2idx):
        weights_matrix_labels.append(word)
        try:
            weights_matrix[i] = torch.tensor(w2v_model[word])
            words_found += 1
        except KeyError:
            weights_matrix[i] = torch.tensor(
                np.random.normal(scale=0.6, size=(emb_dim, )))


    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that
    # the dependence of e. g. 'g' on 'f' can not be learned, but allows more
    # efficient batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)


    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###########################################################################
    # Build the model
    ###########################################################################

    model = RNNModel(model_type,
                     weights_matrix,
                     nhid,
                     nlayers,
                     dropout,
                     tied).to(device)

    print('Create tensorboardX writer')
    tb_writer = SummaryWriter(comment='test')
    print('tensorboard: logging initial embedding weights')
    tb_writer.add_embedding(mat=model.encoder.weight,
                            metadata=weights_matrix_labels)
    save_path = tb_writer.file_writer.get_logdir() + '/' + 'model.pt'
    print(f'all data saving to saving to {tb_writer.file_writer.get_logdir()}')

    criterion = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ###########################################################################
    # Training code
    ###########################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        record_loss = 0
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was
            # previously produced.
            # If we didn't, the model would try backpropagating all the way to
            # start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()   # zero the gradient buffers
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()    # Does the update
            total_loss += loss.item()
            record_loss += loss.item()
            tb_writer.add_scalar('training_loss', loss.item())

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| ms/batch {:5.2f} '
                      '| loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt,
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


    # Loop over epochs.
    best_val_loss = None

    valid_losses = []

    print(f'Starting training run...')
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(0, epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            tb_writer.add_scalar('validation_loss', val_loss)
            valid_losses.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:5.2f} | valid ppl {:8.2f}'
                  ''.format(epoch,
                            (time.time() - epoch_start_time),
                            val_loss,
                            math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen
            # so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save_path, 'wb') as f:
                    torch.save(model, f)
                ## Save State Dictionary
                with open(save_path+'_state', 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


    # Load the best saved model.
    with open(save_path, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    tb_writer.add_scalar('test_loss', test_loss)
    tb_writer.add_embedding(mat=model.encoder.weight,
                            metadata=weights_matrix_labels)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':
    run()
