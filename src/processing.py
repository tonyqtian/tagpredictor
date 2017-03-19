'''
Created on Mar 17, 2017

@author: tonyq
'''
import logging
from keras.callbacks import EarlyStopping
from utils import setLogger, mkdir
from data_processing import get_pdTable, tableMerge, tokenizeIt, word2num, to_categorical2D
from model_processing import getModel, makeEmbedding
from model_eval import Evaluator

logger = logging.getLogger(__name__)

def train(args):
	output_dir = args.out_dir_path
	mkdir(output_dir)
	timestr = setLogger(out_dir=output_dir)
	
	# process train and test data
	_, train_title, train_content, train_tag = get_pdTable(args.train_path)
	_, test_title, test_content, test_tag = get_pdTable(args.test_path)
	
	train_body = tableMerge([train_title, train_content])
	test_body = tableMerge([test_title, test_content])
	
	train_body, train_maxLen = tokenizeIt(train_body, clean=True)
	test_body, test_maxLen = tokenizeIt(test_body, clean=True)
	train_tag, _ = tokenizeIt(train_tag)
	test_tag, _ = tokenizeIt(test_tag)
	maxInputLength = max(train_maxLen, test_maxLen)
	outputLength = 5
	
	embdw2v, vocabDict, vocabReverseDict = makeEmbedding(args, [train_body, test_body])
	# vocabDict, vocabReverseDict, maxInputLength = createVocab([train_body, test_body])
	# logger.info(vocabDict)
	# logger.info(vocabReverseDict)
	
	# word to padded numerical np array
	train_x = word2num(train_body, vocabDict, maxInputLength)
	test_x = word2num(test_body, vocabDict, maxInputLength)
	train_y = word2num(train_tag, vocabDict, outputLength, postpad=True)
	train_y = to_categorical2D(train_y, len(vocabDict))
	test_y = word2num(test_tag, vocabDict, outputLength, postpad=True)
	test_y = to_categorical2D(test_y, len(vocabDict))
	
	# create model 
	rnnmodel = getModel(maxInputLength, outputLength, len(vocabDict), embd=embdw2v, embd_dim=args.embd_dim, rnn_opt=args.rnn_opt)
	rnnmodel.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['categorical_accuracy'])
	rnnmodel.summary()
	
	# train and test model
	evl = Evaluator(args, output_dir, timestr, 'categorical_accuracy', test_x, test_y)
	earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto')
	rnnmodel.fit(train_x, train_y, validation_split=args.valid_split, batch_size=args.train_batch_size, nb_epoch=args.epochs, callbacks=[earlystop, evl])
	# rnnmodel.fit(train_x, train_y, validation_split=0.2, batch_size=8, nb_epoch=2)
