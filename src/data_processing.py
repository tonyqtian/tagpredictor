'''
Created on Mar 17, 2017

@author: tonyq
'''

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import logging
from keras.preprocessing.sequence import pad_sequences
from tqdm._tqdm import tqdm

logger = logging.getLogger(__name__)

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
	if x:
		# BeautifulSoup on content
		soup = BeautifulSoup(x, "html.parser")
		# Stripping all <code> tags with their content if any
		if soup.code:
			soup.code.decompose()
		# Get all the text out of the html
		text = soup.get_text()
		# Returning text stripping out all uris
		return re.sub(uri_re, "", text)
	else:
		return ""
	
def get_words(text):
	word_split = re.compile('[^a-zA-Z0-9_\\+\\-]')
	return [word.strip().lower() for word in word_split.split(text)]
	
def get_pdTable(path):
	logger.info(' Processing pandas csv ')
	pdtable = pd.read_csv(path)
	return pdtable.id, pdtable.title, pdtable.content, pdtable.tags

def tableMerge(tableList):
	return [' '.join(str1) for str1 in zip(*tableList)]

def tokenizeIt(table, clean=False):
	tokenizedTable = []
	maxLen = 0
	for content in tqdm(table):
		if clean:
			text = stripTagsAndUris(content)
			text = get_words(text)
			tokenizedTable.append(text)
			if len(text) > maxLen:
				maxLen = len(text)
		else:
			text = content.split(' ')
			tokenizedTable.append(text)
			if len(text) > maxLen:
				maxLen = len(text)
	return tokenizedTable, maxLen		
	
def createVocab(tableList):
	logger.info(' Creating vocabulary ')
	contentList = []
	for list1 in tableList:
		contentList.extend(list1)
	maxLen = 0
	wdFrq = {}
	total_words = 0
	for line in contentList:
		if len(line) > maxLen:
			maxLen = len(line)
		for wd in line:
			try:
				wdFrq[wd] += 1
			except KeyError:
				wdFrq[wd] = 1
			total_words += 1
	logger.info('  %i total words, %i unique words ' % (total_words, len(wdFrq)))
	import operator
	sorted_word_freqs = sorted(wdFrq.items(), key=operator.itemgetter(1), reverse=True)
	vocab_size = 0
	for _, freq in sorted_word_freqs:
		if freq > 1:
			vocab_size += 1
	vocabDict = {'<pad>':0, '<unk>':1}
	vocabReverseDict = ['<pad>', '<unk>']
	vocabLen = len(vocabDict)
	index = vocabLen	
	for word, _ in sorted_word_freqs[:vocab_size - vocabLen]:
		vocabDict[word] = index
		index += 1
		vocabReverseDict.append(word)
		
	logger.info('  vocab size %i ' % len(vocabReverseDict))
	return vocabDict, vocabReverseDict, maxLen

def word2num(contentTable, vocab, maxLen):
	unk_hit = 0
	totalword = 0
	data = []
	for line in contentTable:
		w2num = []
		for word in line:			
			if word in vocab:
				w2num.append(vocab[word])
			else:
				w2num.append(vocab[''])
				unk_hit += 1
			totalword += 1
		data.append(w2num)
	# pad to np array	
	np_ary = pad_sequences(data, maxlen=maxLen)
	return np_ary

def to_categorical2D(y, nb_classes=None):
	if not nb_classes:
		nb_classes = y.max()
	return (np.arange(nb_classes) == y[:,:,None]-1).astype(int)