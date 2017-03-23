'''
Created on Mar 17, 2017

@author: tonyq
'''
import matplotlib
matplotlib.use('Agg')
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.WARN)

import logging, time
from keras.callbacks import EarlyStopping
from util.utils import setLogger, mkdir
from util.model_eval import Evaluator
from util.w2v_embedding import makeEmbedding
from util.data_processing import get_pdTable, tableMerge, tokenizeIt, createVocab, word2num, to_categorical2D, to_categoricalAll

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
	if args.model == 'seq2seq':
		
		outputLength = 5 + 1	
		use_argm = True
		if args.w2v:
			embdw2v, vocabDict, vocabReverseDict = makeEmbedding(args, [train_body, test_body])
			unk = None
			eof = None
		else:
			vocabDict, vocabReverseDict = createVocab([train_body, test_body], min_count=3, reservedList=['<pad>', '<EOF>', '<unk>'])
			embdw2v = None
			unk = '<unk>'
			eof = '<EOF>'
		pred_vocabDict, pred_vocabReverseDict = createVocab([train_tag,], min_count=3, reservedList=['<pad>', '<EOF>', '<unk>'])
		# logger.info(vocabDict)
		logger.info(pred_vocabReverseDict)		
		# word to padded numerical np array
		train_x = word2num(train_body, vocabDict, unk, inputLength, padding='pre',eof=eof)
		test_x = word2num(test_body, vocabDict, unk, inputLength, padding='pre', eof=eof)
		train_y = word2num(train_tag, pred_vocabDict, unk, outputLength, padding='post', eof=eof)
		train_y = to_categorical2D(train_y, len(pred_vocabDict))
		test_y = word2num(test_tag, pred_vocabDict, unk, outputLength, padding='post', eof=eof)
		test_y = to_categorical2D(test_y, len(pred_vocabDict))

		# choose model 
		from src.seq2seq_model import getModel

	elif args.model == 'categ':
		outputLength = 1
		use_argm = False
		
		if args.w2v:
			embdw2v, vocabDict, vocabReverseDict = makeEmbedding(args, [train_body, test_body])
			unk = None
		else:
			vocabDict, vocabReverseDict = createVocab([train_body, test_body], min_count=3, reservedList=['<pad>', '<unk>'])
			embdw2v = None
			unk = '<unk>'
		pred_vocabDict, pred_vocabReverseDict = createVocab([train_tag,], min_count=3, reservedList=[])
		pred_unk = None
		# logger.info(vocabDict)
		logger.info(pred_vocabReverseDict)
		
		# word to padded numerical np array
		train_x = word2num(train_body, vocabDict, unk, inputLength, padding='pre')
		test_x = word2num(test_body, vocabDict, unk, inputLength, padding='pre')
		train_y = word2num(train_tag, pred_vocabDict, pred_unk, outputLength)
		train_y = to_categoricalAll(train_y, len(pred_vocabDict))
		test_y = word2num(test_tag, pred_vocabDict, pred_unk, outputLength)
		test_y = to_categoricalAll(test_y, len(pred_vocabDict))
		
		# choose model 
		from src.categ_model import getModel
		
	rnnmodel = getModel(args, inputLength, outputLength, len(vocabDict), len(pred_vocabDict), embd=embdw2v)

	if args.optimizer == 'rmsprop':
		from keras.optimizers import RMSprop
		optimizer = RMSprop(lr=args.learning_rate)
	else:
		optimizer = args.optimizer

	if args.loss == 'my_binary_crossentropy':
		from util.my_optimizer import my_binary_crossentropy
		loss = my_binary_crossentropy
	else:
		loss = args.loss

	myMetrics = 'fmeasure'
	rnnmodel.compile(loss=loss, optimizer=optimizer, metrics=[myMetrics])
	rnnmodel.summary()

	if args.save_model:
		## Plotting model
		logger.info('Plotting model architecture')
		from keras.utils.visualize_util import plot	
		plot(rnnmodel, to_file = output_dir + '/' + timestr + 'model_plot.png')
		logger.info('  Done')
			
		## Save model architecture
		logger.info('Saving model architecture')
		with open(output_dir + '/'+ timestr + 'model_config.json', 'w') as arch:
			arch.write(rnnmodel.to_json(indent=2))
		logger.info('  Done')
	
	# train and test model
	myCallbacks = []
	if args.eval_on_epoch:
		evl = Evaluator(args, output_dir, timestr, myMetrics, test_x, test_y, vocabReverseDict, pred_vocabReverseDict, use_argm=use_argm)
		myCallbacks.append(evl)
	if args.earlystop:
		earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto')
		myCallbacks.append(earlystop)
	rnnmodel.fit(train_x, train_y, validation_split=args.valid_split, batch_size=args.train_batch_size, nb_epoch=args.epochs, callbacks=myCallbacks)
	if not args.eval_on_epoch:
		rnnmodel.evaluate(test_x, test_y, batch_size=args.eval_batch_size)
	
	# test output (remove duplicate, remove <pad> <unk>, comparable layout, into csv)
	# final inference: output(remove duplicate, remove <pad> <unk>, limit output words to 3 or 2 or 1..., into csv)