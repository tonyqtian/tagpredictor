'''
Created on Mar 19, 2017

@author: tonyq
'''

import argparse
from time import sleep

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("--train-path", dest="train_path", type=str, metavar='<str>', default='data/train_s4.tsv', help="The path to the training set")
parser.add_argument("--test-path", dest="test_path", type=str, metavar='<str>', default='data/test_s4.tsv', help="The path to the test set")
parser.add_argument("--out-dir", dest="out_dir_path", type=str, metavar='<str>', default='output', help="The path to the output directory")
parser.add_argument("--model-type", dest="model_type", type=str, metavar='<str>', default='cls', help="Model type (cls|reg) (default=cls)")
parser.add_argument("--optimizer", dest="optimizer", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae|cnp|hng) (default=mse) set to cnp if cls model")
parser.add_argument("--embding-dim", dest="embd_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("--cnn-kernel", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=0, help="RNN dimension. '0' means no RNN layer (default=0)")
parser.add_argument("--train-batch-size", dest="train_batch_size", type=int, metavar='<int>', default=8, help="Train Batch size (default=8)")
parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, metavar='<int>', default=8, help="Eval Batch size (default=8)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.4, help="The dropout probability. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-w", dest="dropout_w", type=float, metavar='<float>', default=0.0, help="The dropout probability of RNN W. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-u", dest="dropout_u", type=float, metavar='<float>', default=0.0, help="The dropout probability of RNN U. To disable, give a negative number (default=0.4)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', default=None, help="(Optional) Path to the existing vocab file (*.pkl)")
parser.add_argument("--embedding-path", dest="emb_path", type=str, metavar='<str>', default=None, help="(Optional) Path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=1, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1111, help="Random seed (default=1234)")
parser.add_argument("--plot", dest="plot", action='store_true', help="Save PNG plot")
parser.add_argument("--onscreen", dest="onscreen", action='store_true', help="Show log on stdout")
parser.add_argument("--earlystop", dest="earlystop", type=int, metavar='<int>', default=4, help="Use early stop")
parser.add_argument("--verbose", dest="verbose", type=int, metavar='<int>', default=1, help="Show training process bar during train and val")
parser.add_argument("--valid-split", dest="valid_split", type=float, metavar='<float>', default=0.2, help="Split validation set from training set (default=0.0)")
parser.add_argument("--rnn-optimization", dest="rnn_opt", type=str, metavar='<str>', default='gpu', help="RNN consume_less (cpu|mem|gpu) (default=gpu)")
args = parser.parse_args()

from processing import train
train(args)

print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
