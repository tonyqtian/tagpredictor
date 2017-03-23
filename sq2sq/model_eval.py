'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from numpy import argmax
from util.eval_metrics import f1_score_prec_rec

logger = logging.getLogger(__name__)

class Evaluator(Callback):
	
	def __init__(self, args, out_dir, timestr, metric, test_x, test_y, reVocab, pred_reVocab):
		self.out_dir = out_dir
		self.test_x = test_x
		self.test_y = test_y
		self.best_test = -1
		self.best_epoch = -1
		self.batch_size = args.eval_batch_size
		self.metric = metric
		self.val_metric = 'val_' + metric
		self.timestr = timestr
		self.losses = []
		self.accs = []
		self.val_accs = []
		self.val_losses = []
		self.test_losses = []
		self.test_accs = []
		self.test_precisions = []
		self.test_recalls = []
		self.test_f1s = []
		self.plot = args.plot
		self.evl_pred = args.show_evl_pred
		self.reVocab = reVocab
		self.pred_reVocab = pred_reVocab
		
	def eval(self, model, epoch, print_info=False):
# 		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size)
# 		self.test_losses.append(self.test_loss)
# 		self.test_accs.append(self.test_metric)
		if self.evl_pred:
			pred = model.predict(self.test_x, batch_size=self.batch_size)
			preds = argmax(pred, axis=-1)
			reals = argmax(self.test_y, axis=-1)
			precision, recall, f1_score = f1_score_prec_rec(reals, preds, make_unique=True, remove_pad=[0,1])
			self.test_f1s.append(f1_score)
			self.test_recalls.append(recall)
			self.test_precisions.append(precision)
			self.print_pred(self.test_x[:self.evl_pred], preds[:self.evl_pred], reals[:self.evl_pred])
			self.print_info(epoch, precision, recall, f1_score)

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accs.append(logs.get(self.metric))
		self.val_accs.append(logs.get(self.val_metric))
		self.eval(self.model, epoch, print_info=True)
		if self.plot:
			self.plothem()
		return

	def plothem(self):
		training_epochs = [i for i in range(len(self.losses))]
		plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
		plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
		plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
		plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
# 		plt.plot(training_epochs, self.test_losses, 'k', label='Test Loss')
# 		plt.plot(training_epochs, self.test_accs, 'c.', label='Test Metric')
		plt.plot(training_epochs, self.test_f1s, 'k', label='Test F1-Score')
		plt.plot(training_epochs, self.test_precisions, 'c.', label='Test Precision')
		plt.plot(training_epochs, self.test_recalls, 'm.', label='Test Recall')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'LossAccuracy.png')
		plt.close()

	def print_pred(self, infers, preds, reals):
		for (infr, pred, real) in zip(infers, preds, reals):
			infr_line = []
			for strin in infr:
				if not strin == 0:
					infr_line.append(self.reVocab[strin])
			pred_line = [self.pred_reVocab[strin] for strin in pred]
			real_line = [self.pred_reVocab[strin] for strin in real]
			logger.info('[Test]  ')
			logger.info('[Test]  Line: %s ' % ' '.join(infr_line) )
			logger.info('[Test]  True: %s ' % ' '.join(real_line) )
			logger.info('[Test]  Pred: %s ' % ' '.join(pred_line) )
							
	def print_info(self, epoch, precision, recall, f1score):
		logger.info('[Test]  Epoch: %i' % epoch)
		logger.info('[Test]  F1-Score: %.4f, Precision: %.4f, Recall: %.4f' % (f1score, precision, recall))
