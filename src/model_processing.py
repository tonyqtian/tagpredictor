'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dense, RepeatVector
from gensim.models.word2vec import Word2Vec
# import keras.backend as K

logger = logging.getLogger(__name__)

def getModel(input_length, output_length, vocab_size, embd, embd_dim, embd_trainable=True, rnn_opt='cpu', rnn_dim=32):
# 	if embd == None:
# 		my_init = 'uniform'
# 		logger.info(' Use default initializing embedding')
# 	else:
# 		if embd.shape == (vocab_size, embd_dim):
# 			def my_init(embd, name=None):
# 				return K.variable(embd, name=name)
# 			logger.info(' Use pre-trained embedding')
# 		else:
# 			raise ValueError('Incorrect embedding shape', embd.shape, ' expect ', (vocab_size, embd_dim))
		
	# encoder
	sequence = Input(shape=(input_length,), dtype='int32')
	x = Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable)(sequence)
# 	x = LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt)(x)
	x = Bidirectional(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt))(x)
	x = Bidirectional(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt))(x)
	x = Dense(rnn_dim*2, activation='relu')(x)
	
	# decoder
	pred = RepeatVector(output_length)(x)
	pred = LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt)(pred)
	pred = TimeDistributed(Dense(vocab_size, activation='softmax'))(pred)
	
	model = Model(input=sequence, output=pred)
	return model

def makeEmbedding(inputTable):
	sentenceList = []
	for tbl in inputTable:
		sentenceList.extend(tbl)
	print(len(inputTable))
	print(len(inputTable[0]))
	print(len(inputTable[1]))
	print(len(sentenceList))
	
	class SentenceGenerator(object):
		def __init__(self, sentList):
			self.sentList = sentList
		
		def __iter__(self):
			for line in self.sentList:
				yield line
				
	sentences = SentenceGenerator(sentenceList)
	w2vModel = Word2Vec(sentences, min_count=2, size=100)
# 	w2vModel.save('../data/embd_model.bin')
	embdWeights = w2vModel.wv.syn0
	print(embdWeights.shape)
	
	vocabDict = dict([(k, v.index) for k, v in w2vModel.wv.vocab.items()])
	logger.info('  Vocabulary size %i ' % (len(vocabDict)))
# 	print(vocabDict)
	import operator
	sorted_word = sorted(vocabDict.items(), key=operator.itemgetter(1), reverse=False)
	vocabReverseDict = []
	for word, _ in sorted_word:
		vocabReverseDict.append(word)
# 	print(vocabReverseDict)
	# eval will take hours without word limiting
# 	w2vModel.accuracy('../data/questions-words.txt')
	del w2vModel
	return embdWeights, vocabDict, vocabReverseDict