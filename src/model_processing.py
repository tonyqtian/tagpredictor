'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Model, Sequential
from keras.engine.topology import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dense, RepeatVector, Activation, Dropout
from gensim.models.word2vec import Word2Vec
# from attention_wrapper import Attention

logger = logging.getLogger(__name__)

def getModel(input_length, output_length, vocab_size, pred_size, embd, embd_dim, embd_trainable=True, rnn_opt='cpu', rnn_dim=32):
		
	sequence = Input(shape=(input_length,), dtype='int32')
	if type(embd) is type(None):
		x = Embedding(vocab_size, embd_dim, mask_zero=True, trainable=embd_trainable)(sequence)
	else:
		x = Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable)(sequence)
	
# 	model = Sequential()
# 	if type(embd) is type(None):
# 		model.add(Embedding(vocab_size, embd_dim, mask_zero=False, trainable=embd_trainable, batch_input_shape=(None, input_length)))
# 	else:
# 		model.add(Embedding(vocab_size, embd_dim, mask_zero=False, weights=[embd], trainable=embd_trainable, batch_input_shape=(None, input_length)))

	# encoder
# 	x = Bidirectional(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt))(x)
# 	x = Bidirectional(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt))(x)
	x = LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt)(x)
	x = LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt)(x)
	x = Activation('relu')(x)
# 	x = Dense(rnn_dim, activation='relu')(x)
	
	# decoder
	pred = RepeatVector(output_length)(x)
	pred = LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt)(pred)
	pred = TimeDistributed(Dense(pred_size, activation='softmax'))(pred)
# 	model.add(Activation('softmax'))
	model = Model(input=sequence, output=pred)
	return model

def makeEmbedding(args, inputTable):
	sentenceList = []
	for tbl in inputTable:
		sentenceList.extend(tbl)
	logger.info('  Total %i lines info for word2vec processing ' % (len(sentenceList)))
	
	class SentenceGenerator(object):
		def __init__(self, sentList):
			self.sentList = sentList
		
		def __iter__(self):
			for line in self.sentList:
				yield line
				
	sentences = SentenceGenerator(sentenceList)
	w2vModel = Word2Vec(sentences, min_count=2, size=args.embd_dim)
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