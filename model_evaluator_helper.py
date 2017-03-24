'''
Created on Mar 24, 2017

@author: tonyq
'''

import argparse
from time import sleep

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("--model-prama-path", dest="prama_path", type=str, metavar='<str>')
parser.add_argument("--weight-prama-path", dest="weight_path", type=str, metavar='<str>')
parser.add_argument("--infer-path", dest="infer_path", type=str, metavar='<str>')
parser.add_argument("--standard-path", dest="standard_path", type=str, default=None, metavar='<str>')
parser.add_argument("--vocab-path", dest="vocab_path", type=str, default=None, metavar='<str>')
parser.add_argument("--output-path", dest="output_path", type=str, default=None, metavar='<str>')
parser.add_argument("--model", dest="model", type=str, metavar='<str>', default='seq2seq', help="Model Type: seq2seq, categ")
parser.add_argument("--optimizer", dest="optimizer", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--loss", dest="loss", type=str, metavar='<str>', default='categorical_crossentropy', help="Loss function")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar='<float>', default=0.0001, help="Optimizer learning rate")
parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, metavar='<int>', default=8, help="Eval Batch size (default=8)")
parser.add_argument("--max-tag", dest="max_tag", type=int, metavar='<int>', default=4, help="Max tag for each entry")
args = parser.parse_args()

from src.model_evaluator import evaluator
evaluator(args)

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
