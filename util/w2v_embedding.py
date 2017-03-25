'''
Created on Mar 24, 2017

@author: tonyq
'''
import logging
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)

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
	w2vModel = Word2Vec(sentences, min_count=2, size=args.embd_dim, null_word=1)
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
