'''
Created on Mar 17, 2017

@author: tonyq
'''
import matplotlib
from keras.backend.tensorflow_backend import categorical_crossentropy
matplotlib.use('Agg')

import logging, time
from keras.callbacks import EarlyStopping
from src.utils import setLogger, mkdir
from categ.data_processing import get_pdTable, tableMerge, tokenizeIt, createVocab, word2num, to_categoricalAll
from categ.model_processing import getModel, makeEmbedding
from categ.model_eval import Evaluator

logger = logging.getLogger(__name__)

def train(args):
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	mkdir(output_dir)
	setLogger(timestr, out_dir=output_dir)
	
	# process train and test data
	_, train_title, train_content, train_tag = get_pdTable(args.train_path)
	_, test_title, test_content, test_tag = get_pdTable(args.test_path)
	
	train_body = tableMerge([train_title, train_content])
	test_body = tableMerge([test_title, test_content])
	
	train_body, _ = tokenizeIt(train_body, clean=True)
	test_body, _ = tokenizeIt(test_body, clean=True)
	train_tag, _ = tokenizeIt(train_tag)
	test_tag, _ = tokenizeIt(test_tag)
# 	inputLength = max(train_maxLen, test_maxLen)
	inputLength = 400
	outputLength = 1
	
	if args.w2v:
		embdw2v, vocabDict, vocabReverseDict = makeEmbedding(args, [train_body, test_body])
		unk = None
	else:
		vocabDict, vocabReverseDict = createVocab([train_body, test_body], min_count=2)
		print(vocabReverseDict)
		embdw2v = None
		unk = '<unk>'
	pred_vocabDict, pred_vocabReverseDict = createVocab([train_tag,], min_count=1)
	# logger.info(vocabDict)
	logger.info(pred_vocabReverseDict)
	if args.attention:
		mypad = 'pre'
	else:
		mypad = None
	# word to padded numerical np array
	train_x = word2num(train_body, vocabDict, unk, inputLength, padding=mypad)
	test_x = word2num(test_body, vocabDict, unk, inputLength, padding=mypad)
	train_y = word2num(train_tag, pred_vocabDict, unk, outputLength)
	train_y = to_categoricalAll(train_y, len(pred_vocabDict))
	test_y = word2num(test_tag, pred_vocabDict, unk, outputLength)
	test_y = to_categoricalAll(test_y, len(pred_vocabDict))
	# create model 
	rnnmodel = getModel(args, inputLength, outputLength, len(vocabDict), len(pred_vocabDict), embd=embdw2v)
	from keras.optimizers import RMSprop
	optimizer = RMSprop(lr=args.learning_rate)
	myMetrics = 'fmeasure'
	rnnmodel.compile(loss=args.loss, optimizer=optimizer, metrics=[myMetrics])
	rnnmodel.summary()
	
	# train and test model
	myCallbacks = []
	if args.eval_on_epoch:
		evl = Evaluator(args, output_dir, timestr, myMetrics, test_x, test_y, vocabReverseDict, pred_vocabReverseDict)
		myCallbacks.append(evl)
	if args.earlystop:
		earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto')
		myCallbacks.append(earlystop)
	rnnmodel.fit(train_x, train_y, validation_split=args.valid_split, batch_size=args.train_batch_size, nb_epoch=args.epochs, callbacks=myCallbacks)
	if not args.eval_on_epoch:
		rnnmodel.evaluate(test_x, test_y, batch_size=args.eval_batch_size)
	
	# test output (remove duplicate, remove <pad> <unk>, comparable layout, into csv)
	# final inference: output(remove duplicate, remove <pad> <unk>, limit output words to 3 or 2 or 1..., into csv)