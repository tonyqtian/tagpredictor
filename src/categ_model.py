'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Activation, Dropout
from util.attention_wrapper import Attention
from util.my_layers import DenseWithMasking
from keras.layers.wrappers import Bidirectional

logger = logging.getLogger(__name__)

def getModel(args, input_length, output_length, vocab_size, pred_size, embd, embd_trainable=True):
	embd_dim = args.embd_dim
	rnn_opt = args.rnn_opt
	rnn_dim = args.rnn_dim
	dropout_W = args.dropout_w
	dropout_U = args.dropout_u
	if args.activation == 'sigmoid':
		final_init = 'he_normal'
	else:
		final_init = 'he_uniform'
	
	model = Sequential()
	if type(embd) is type(None):
		model.add(Embedding(vocab_size, embd_dim, mask_zero=True, trainable=embd_trainable, batch_input_shape=(None, input_length)))
	else:
		model.add(Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable, batch_input_shape=(None, input_length)))
	
	for _ in range(args.rnn_layer-1):
		if args.bidirectional:
			model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt,
										 dropout_W=dropout_W, dropout_U=dropout_U)))
		else:
			model.add(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt,
						 dropout_W=dropout_W, dropout_U=dropout_U))
		model.add(Dropout(args.dropout_prob))
	if args.attention:			
		model.add(Attention(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt,
								 dropout_W=dropout_W, dropout_U=dropout_U)))
	else:
		if args.bidirectional:
			model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt,
										 dropout_W=dropout_W, dropout_U=dropout_U)))
		else:
			model.add(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt,
						 dropout_W=dropout_W, dropout_U=dropout_U))
	model.add(Dropout(args.dropout_prob))
	model.add(DenseWithMasking(pred_size, init=final_init))
	model.add(Activation(args.activation))
	return model
