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
parser.add_argument("--train-path", dest="train_path", type=str, metavar='<str>', default='../data/robotics.csv', help="The path to the training set")
parser.add_argument("--test-path", dest="test_path", type=str, metavar='<str>', default='../data/physics_sample.csv', help="The path to the test set")
parser.add_argument("--out-dir", dest="out_dir_path", type=str, metavar='<str>', default='output', help="The path to the output directory")
parser.add_argument("--optimizer", dest="optimizer", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--loss", dest="loss", type=str, metavar='<str>', default='categorical_crossentropy', help="Loss function")
parser.add_argument("--activation", dest="activation", type=str, metavar='<str>', default='softmax', help="Activation function")
parser.add_argument("--embedding-dim", dest="embd_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("--cnn-kernel", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("--cnn-win", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("--rnn-dim", dest="rnn_dim", type=int, metavar='<int>', default=4, help="RNN dimension (default=4)")
parser.add_argument("--rnn-layer", dest="rnn_layer", type=int, metavar='<int>', default=2, help="RNN layers (default=2)")
parser.add_argument("--train-batch-size", dest="train_batch_size", type=int, metavar='<int>', default=8, help="Train Batch size (default=8)")
parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, metavar='<int>', default=8, help="Eval Batch size (default=8)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.4, help="The dropout probability. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-w", dest="dropout_w", type=float, metavar='<float>', default=0.3, help="The dropout probability of RNN W. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-u", dest="dropout_u", type=float, metavar='<float>', default=0.3, help="The dropout probability of RNN U. To disable, give a negative number (default=0.4)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', default=None, help="(Optional) Path to the existing vocab file (*.pkl)")
parser.add_argument("--embedding-path", dest="emb_path", type=str, metavar='<str>', default=None, help="(Optional) Path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=1, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1111, help="Random seed (default=1234)")
parser.add_argument("--plot", dest="plot", action='store_true', help="Save PNG plot")
parser.add_argument("--onscreen", dest="onscreen", action='store_true', help="Show log on stdout")
parser.add_argument("--earlystop", dest="earlystop", type=int, metavar='<int>', default=4, help="Use early stop")
parser.add_argument("--verbose", dest="verbose", type=int, metavar='<int>', default=1, help="Show training process bar during train and val")
parser.add_argument("--valid-split", dest="valid_split", type=float, metavar='<float>', default=0.1, help="Split validation set from training set (default=0.0)")
parser.add_argument("--mem-opt", dest="rnn_opt", type=str, metavar='<str>', default='gpu', help="RNN consume_less (cpu|mem|gpu) (default=gpu)")
parser.add_argument("--eval-on-epoch", dest="eval_on_epoch", action='store_true', help="Test after every epoch")
parser.add_argument("--show-eval-pred", dest="show_evl_pred", type=int, metavar='<int>', default=0, help="Show <num> predicts after every test pred")
parser.add_argument("--w2v-embedding", dest="w2v", action='store_true', help="Use pre-trained word2vec embedding")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar='<float>', default=0.01, help="Optimizer learning rate")
parser.add_argument("--seq2seq", dest="seq2seq", type=int, metavar='<int>', default=0, help="Use Seq2Seq Model")
parser.add_argument("--attention", dest="attention", action='store_true', help="Use Attention Wrapper")
parser.add_argument("--save-model", dest="save_model", action='store_true', help="Save Model Parameters")
parser.add_argument("--model", dest="model", type=str, metavar='<str>', default='seq2seq', help="Model Type: seq2seq, categ")
parser.add_argument("--bi-directional", dest="bidirectional", action='store_true', help="Use Bi-directional RNN")
args = parser.parse_args()

from src.processing import train
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
