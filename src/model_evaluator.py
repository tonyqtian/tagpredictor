'''
Created on Mar 24, 2017

@author: tonyq
'''
from keras.models import model_from_json
import pickle as pkl
from numpy import argmax
from util.eval_metrics import f1_score_prec_rec
from util.data_processing import get_pdTable, tableMerge, tokenizeIt, word2num, prob_top_n
from util.my_layers import DenseWithMasking

def evaluator(args):
	
	# process train and test data
	print("Processing Inference File ...")
	_, test_title, test_content = get_pdTable(args.infer_path, notag=True)
	test_body = tableMerge([test_title, test_content])	
	test_body, _ = tokenizeIt(test_body, clean=True)
	if args.standard_path:
		print("Processing Standard File ...")
		_, _, _, standard_tag = get_pdTable(args.standard_path)
		standard_tag, _ = tokenizeIt(standard_tag)
	inputLength = 400
	
	with open(args.vocab_path, 'rb') as vocab_file:
		(vocabDict, vocabReverseDict, pred_vocabDict, pred_vocabReverseDict) = pkl.load(vocab_file)

	if args.model == 'seq2seq':
		unk = '<unk>'
		eof = '<EOF>'
		outputLength = 5 + 1	
		# word to padded numerical np array
		test_x = word2num(test_body, vocabDict, unk, inputLength, padding='pre', eof=eof)
		if args.standard_path:
			reals = word2num(standard_tag, pred_vocabDict, unk, outputLength, padding='post', eof=eof)

	elif args.model == 'categ':
		unk = None
		pred_unk = None
		outputLength = 1
		# word to padded numerical np array
		test_x = word2num(test_body, vocabDict, unk, inputLength, padding='pre')
		if args.standard_path:
			reals = word2num(standard_tag, pred_vocabDict, pred_unk, outputLength)
		
	else:
		raise NotImplementedError	
		
	# load json and create model
	with open(args.prama_path, 'r') as json_file:
		loaded_model_json = json_file.read()
		loaded_model = model_from_json(loaded_model_json, custom_objects={"DenseWithMasking": DenseWithMasking})
		# load weights into new model
	loaded_model.load_weights(args.weight_path)
	print("Loaded model from disk")
	
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
	loaded_model.compile(loss=loss, optimizer=optimizer, metrics=[myMetrics])
	loaded_model.summary()
	
	# inference loaded model on test data
	pred = loaded_model.predict(test_x, batch_size=args.eval_batch_size)
	print("Prediction Finished")
	if args.model == 'seq2seq':
		preds = argmax(pred, axis=-1)
		if args.standard_path:
			precision, recall, f1_score = f1_score_prec_rec(reals, preds, make_unique=True, remove_pad=[0,1])
	elif args.model == 'categ':
		preds = prob_top_n(pred, top=5)
		if args.standard_path:
			precision, recall, f1_score = f1_score_prec_rec(reals, preds)
	else:
		raise NotImplementedError
	print("Pred calculation finished")
	if args.standard_path:
		print('F1-Score: %.4f, Precision: %.4f, Recall: %.4f' % (f1_score, precision, recall))
	
	if args.output_path:
		max_tags = args.max_tag
		import csv
		with open(args.output_path, 'w') as output_fhdl:
			writer = csv.writer(output_fhdl)
			writer.writerow(['id','tags'])
			rowid = 1
			for row in preds:
				row = row[:max_tags]
				pred_tag = [pred_vocabReverseDict[idx] for idx in row]
				writer.writerow([rowid, " ".join(pred_tag)])
				rowid += 1