import random
import sys
import logging
import click
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.onnx
from gensim.models import KeyedVectors

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler

from word2vec_lstm.data import Corpus, RecipesDataset
from word2vec_lstm.model import RNNModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option('--data_path', type=str,
              help='location of the data corpus')
@click.option('--model_type', type=str, default='LSTM',
              help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
@click.option('--nhid', type=int, default=300,
              help='number of hidden units per layer')
@click.option('--nlayers', type=int, default=2,
              help='number of layers')
@click.option('--lr', type=float, default=0.0001,
              help='initial adam learning rate')
@click.option('--clip', type=float, default=0.25,
              help='gradient clipping')
@click.option('--epochs', type=int, default=70,
              help='upper epoch limit')
@click.option('--batch_size', type=int, default=20, metavar='N',
              help='batch size')
@click.option('--bptt', type=int, default=100,
              help='sequence length')
@click.option('--dropout', type=float, default=0.4,
              help='dropout applied to layers (0 = no dropout)')
@click.option('--tied', default=True,
              help='tie the word embedding and softmax weights')
@click.option('--seed', type=int, default=1111,
              help='random seed')
@click.option('--log-interval', type=int, default=50,
              help='report interval')
@click.option('--log-level', type=str, default='info',
              help='info, debug, warn, etc')
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
        log_interval: int,
        log_level: str):

    if log_level.lower() == 'debug':
        logger.setLevel(logging.DEBUG)
    tb_writer = SummaryWriter(
        comment=f'_nhid{nhid}_nlayers{nlayers}_lr{lr}_clip{clip}'
                f'_bptt{bptt}_dropout{dropout}')
    save_dir = tb_writer.file_writer.get_logdir()
    save_path = save_dir + '/' + 'model.pt'
    fhandler = logging.FileHandler(filename=save_dir + '/out.log')
    logger.addHandler(fhandler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f'all data saving to {save_dir}')

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'using device: {device}')

    ###########################################################################
    # Load data
    ###########################################################################

    corpus = Corpus(data_path, device)

    trainset = RecipesDataset(corpus.train)
    validset = RecipesDataset(corpus.valid)
    testset = RecipesDataset(corpus.test)

    logger.info(f'corpus size: {len(corpus.dictionary)}')

    # Create embedding matrix from corpus
    w2v_path = 'data/GoogleNews-vectors-negative300.bin'
    logger.info(f'reading word2vec trained model from {w2v_path}')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    emb_dim = w2v_model.vector_size

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
    # def batchify(data, bsz):
    #     # Work out how cleanly we can divide the dataset into bsz parts.
    #     nbatch = data.size(0) // bsz
    #     # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #     data = data.narrow(0, 0, nbatch * bsz)
    #     # Evenly divide the data across the bsz batches.
    #     data = data.view(bsz, -1).t().contiguous()
    #     return data.to(device)

    eval_batch_size = 10

    def collate_batch(batch_sz):
        def _collate_batch(batch):
            data = torch.stack(batch).view(-1, batch_sz)
            return data[:-1, :], data[1:, :].view(-1)
        return _collate_batch

    class SkipSampler(Sampler):
        def __init__(self, data_source, batch_size: int, rand: bool):
            super().__init__(data_source)
            self.data_source = data_source
            self._batch_size = batch_size
            self._skip = len(self.data_source) // self._batch_size
            self._rand = rand

        def __iter__(self):
            start = 0
            for i in range(self._skip):
                for j in range(self._batch_size):
                    yield start + j*self._skip + i

        def __len__(self):
            return len(self.data_source)

    train_loader = DataLoader(trainset,
                              batch_sampler=BatchSampler(
                                  SkipSampler(trainset,
                                              batch_size=batch_size,
                                              rand=False),
                                  batch_size=batch_size*(bptt+1),
                                  drop_last=True),
                              collate_fn=collate_batch(batch_size)
                              )
    valid_loader = DataLoader(validset,
                              batch_sampler=BatchSampler(
                                  SkipSampler(validset,
                                              batch_size=eval_batch_size,
                                              rand=False),
                                  batch_size=eval_batch_size*(bptt+1),
                                  drop_last=True),
                              collate_fn=collate_batch(eval_batch_size),
                              )
    test_loader = DataLoader(testset,
                              batch_sampler=BatchSampler(
                                  SkipSampler(testset,
                                              batch_size=eval_batch_size,
                                              rand=False),
                                  batch_size=eval_batch_size*(bptt+1),
                                  drop_last=True),
                              collate_fn=collate_batch(eval_batch_size),
                              )

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.
    #
    # def get_batch(source, i):
    #     seq_len = min(bptt, len(source) - 1 - i)
    #     data = source[i:i + seq_len]
    #     target = source[i + 1:i + 1 + seq_len].view(-1)
    #     return data, target

    ###########################################################################
    # Build the model
    ###########################################################################

    model = RNNModel(model_type,
                     weights_matrix,
                     nhid,
                     nlayers,
                     dropout,
                     tied).to(device)

    logger.info('tensorboard: logging initial embedding weights')
    tb_writer.add_embedding(mat=model.encoder.weight,
                            metadata=weights_matrix_labels,
                            global_step=0)

    criterion = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ###########################################################################
    # Training code
    ###########################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their
        history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            # for i in range(0, data_source.size(0) - 1, bptt):
            #     data, targets = get_batch(data_source, i)
            for data, targets in data_source:
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)

    def train(epoch):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        record_loss = 0
        # for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        #     data, targets = get_batch(train_data, i)
        for i, (data, targets) in enumerate(train_loader):
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
            tb_writer.add_scalar(
                'training_loss',
                loss.item(),
                global_step=epoch*(len(train_loader)) + i)

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| ms/batch {:5.2f} '
                      '| loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_loader),
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    # Loop over epochs.
    best_val_loss = None

    valid_losses = []

    logger.info(f'Starting training run...')
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(0, epochs+1):
            epoch_start_time = time.time()
            train(epoch)
            val_loss = evaluate(valid_loader)
            tb_writer.add_scalar('validation_loss',
                                 val_loss,
                                 global_step=epoch)
            valid_losses.append(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:5.2f} | valid ppl {:8.2f}'
                  ''.format(epoch,
                            (time.time() - epoch_start_time),
                            val_loss,
                            math.exp(val_loss)))
            logger.info('-' * 89)
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
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    # Load the best saved model.
    with open(save_path, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_loader)
    tb_writer.add_scalar('test_loss', test_loss)
    tb_writer.add_embedding(mat=model.encoder.weight,
                            metadata=weights_matrix_labels,
                            global_step=1)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} '
                '| test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    logger.info('=' * 89)


if __name__ == '__main__':
    run()
