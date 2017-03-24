'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dense, RepeatVector, Activation, Dropout
from util.attention_wrapper import Attention
from seq2seq import Seq2Seq

logger = logging.getLogger(__name__)

def getModel(args, input_length, output_length, vocab_size, pred_size, embd, embd_trainable=True):
	embd_dim = args.embd_dim
	rnn_opt = args.rnn_opt
	rnn_dim = args.rnn_dim
	dropout_W = args.dropout_w
	dropout_U = args.dropout_u
	
	model = Sequential()
	if type(embd) is type(None):
		model.add(Embedding(vocab_size, embd_dim, mask_zero=True, trainable=embd_trainable, input_shape=(input_length,)))
	else:
		model.add(Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable, input_shape=(input_length,)))
	
	if args.seq2seq:
		model.add(Seq2Seq(output_dim=pred_size, output_length=output_length, input_shape=(input_length, embd_dim), 
						peek=True, dropout=args.dropout_prob, depth=args.seq2seq))
	else:
		# encoder
		for _ in range(args.rnn_layer-1):
			model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt,
									 	dropout_W=dropout_W, dropout_U=dropout_U)))
			model.add(Dropout(args.dropout_prob))
		if args.attention:
			model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt,
										 dropout_W=dropout_W, dropout_U=dropout_U)))
			model.add(Dropout(args.dropout_prob))
			model.add(TimeDistributed(Dense(rnn_dim)))
		else:
			model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt,
										 dropout_W=dropout_W, dropout_U=dropout_U)))
			model.add(Dropout(args.dropout_prob))
			model.add(Dense(rnn_dim))
		model.add(Activation('relu'))
		# decoder
		if args.attention:
			model.add(Attention(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt,
									 dropout_W=dropout_W, dropout_U=dropout_U)))
		else:
			model.add(RepeatVector(output_length))
			model.add(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt, dropout_W=dropout_W, dropout_U=dropout_U))
		model.add(Dropout(args.dropout_prob))
		model.add(TimeDistributed(Dense(pred_size)))
	model.add(Activation(args.activation))
	return model
