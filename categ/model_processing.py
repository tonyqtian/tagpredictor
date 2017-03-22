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
from gensim.models.word2vec import Word2Vec
from src.attention_wrapper import Attention
from seq2seq import Seq2Seq

logger = logging.getLogger(__name__)

def getModel(args, input_length, output_length, vocab_size, pred_size, embd, embd_trainable=True):
	embd_dim = args.embd_dim
	rnn_opt = args.rnn_opt
	rnn_dim = args.rnn_dim	
# 	sequence = Input(shape=(input_length,), dtype='int32')
# 	if type(embd) is type(None):
# 		x = Embedding(vocab_size, embd_dim, mask_zero=True, trainable=embd_trainable)(sequence)
# 	else:
# 		x = Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable)(sequence)
	
	model = Sequential()
	if type(embd) is type(None):
		model.add(Embedding(vocab_size, embd_dim, mask_zero=False, trainable=embd_trainable, batch_input_shape=(None, input_length)))
	else:
		model.add(Embedding(vocab_size, embd_dim, mask_zero=False, weights=[embd], trainable=embd_trainable, batch_input_shape=(None, input_length)))
	
	if args.seq2seq:
		model.add(Seq2Seq(output_dim=pred_size, output_length=output_length, input_shape=(input_length, embd_dim), peek=True, depth=2))
	else:
		# encoder
		model.add(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt))
		if args.attention:
			model.add(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt))
		else:
			model.add(LSTM(rnn_dim, return_sequences=False, consume_less=rnn_opt))
		model.add(Activation('relu'))
		# decoder
		if args.attention:
			model.add(Attention(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt)))
		else:
			model.add(RepeatVector(output_length))
			model.add(LSTM(rnn_dim, return_sequences=True, consume_less=rnn_opt))
		model.add(TimeDistributed(Dense(pred_size)))
	model.add(Activation(args.activation))
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