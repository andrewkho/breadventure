###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

from .data import Corpus
from .model import RNNModel

parser = argparse.ArgumentParser(description='RNN Recipe generator')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,
                    help='prefix of state-dict')
parser.add_argument('--nemb', type=int, default=650,
                    help='size of embedding layer')
parser.add_argument('--nhid', type=int, default=650,
                    help='hidden size')
parser.add_argument('--nlayers', type=int, default=2,
                    help='num RNN layers')
parser.add_argument('--tie-weights', action='store_true',
                    help='tie encode/decode weights or not')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should "
              "probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)

model = RNNModel(args.model_type, ntokens, args.nemb, args.nhid, args.nlayers,
                 tie_weights=args.tie_weights)
print(f'**** Loading from {model.get_state_filename(args.save)} ****')
with open(model.get_state_filename(args.save), 'rb') as f:
    model.load_state_dict(torch.load(f, map_location='cpu'))
model.eval()

hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
