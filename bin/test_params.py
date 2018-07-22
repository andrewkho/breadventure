import subprocess
import itertools

import click


@click.command()
@click.option('--cmd_path', type=str,
              help='location of the data corpus')
@click.option('--data_path', type=str,
              help='location of the data corpus')
def test_params(cmd_path: str, data_path: str):
    nhid_ = [50, 100, 300]
    nlayers_ = [1, 2, 4]
    lr_ = [0.00001, 0.0001]
    clip_ = [0.2, 0.6]
    epochs = 150
    batch_size_ = [5, 20, 40]
    bptt_ = [35, 70, 300]
    dropout = {50: 0.1, 100: 0.3, 300: 0.5}

    log_level = 'info'

    num_runs = len(nhid_)\
               *len(nlayers_)\
               *len(lr_)\
               *len(clip_)\
               *len(batch_size_)\
               *len(bptt_)
    for i, (nhid, nlayers, lr, clip, batch_size, bptt) \
            in enumerate(itertools.product(nhid_,
                                           nlayers_,
                                           lr_,
                                           clip_,
                                           batch_size_,
                                           bptt_)):
        log_interval = 50000 // (batch_size*bptt)
        if log_interval < 0:
            log_interval = 10
        print(f'Doing run {i}/{num_runs}')
        call = ['python', str(cmd_path),
                '--data_path', str(data_path),
                '--nhid', str(nhid),
                '--nlayers', str(nlayers),
                '--lr', str(lr),
                '--clip', str(clip),
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--bptt', str(bptt),
                '--dropout', str(dropout[nhid] if nlayers > 1 else 0),
                '--tied', str(nhid == 300),
                '--log-interval', str(log_interval),
                '--log-level', str(log_level)]
        print(f'call: {call}')
        subprocess.run(call)


if __name__ == '__main__':
    test_params()
