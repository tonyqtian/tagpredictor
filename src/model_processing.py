'''
Created on Mar 18, 2017

@author: tonyq
'''
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dense, RepeatVector, TimeDistributedDense

def getModel(input_length, output_length, vocab_size, emb_dim=50, embd_init = 'uniform', embd_trainable = True, rnn_opt = 'mem', rnn_dim = 32):
	# encoder
	sequence = Input(shape=(input_length,), dtype='int32')
	x = Embedding(vocab_size, emb_dim, mask_zero=True, init=embd_init, trainable=embd_trainable)(sequence)
	rnn_layer = LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt)
	rnn_layer = Bidirectional(rnn_layer)
	x = rnn_layer(x)
	x = Dense(rnn_dim, activation='relu')(x)
	
	# decoder
	pred = RepeatVector(output_length)(x)
	pred = LSTM(rnn_dim, return_sequences=False)(pred)
	pred = TimeDistributedDense(emb_dim, activation='linear')
	
	model = Model(input=sequence, output=pred)
	return model
