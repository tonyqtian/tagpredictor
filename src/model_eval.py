'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.callbacks import Callback
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Evaluator(Callback):
	
	def __init__(self, out_dir, timestr, metric, test_x, test_y):
		self.out_dir = out_dir
		self.test_x = test_x
		self.test_y = test_y
		self.best_test = -1
		self.best_epoch = -1
		self.batch_size = 1024
		self.metric = metric
		self.val_metric = 'val_' + metric
		self.timestr = timestr
		self.losses = []
		self.accs = []
		self.val_accs = []
		self.val_losses = []
		self.test_losses = []
		self.test_accs = []
		
	def eval(self, model, epoch, print_info=False):
		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size)
		self.test_losses.append(self.test_loss)
		self.test_accs.append(self.test_metric)

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accs.append(logs.get(self.metric))		
		self.val_accs.append(logs.get(self.val_metric))	
		self.eval(self.model, epoch, print_info=True)
# 		self.plothem()
		self.print_info(epoch)
		return

	def plothem(self):
		training_epochs = [i for i in range(len(self.losses))]
		plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
		plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
		plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
		plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
		plt.plot(training_epochs, self.test_losses, 'k', label='Test Loss')
		plt.plot(training_epochs, self.test_accs, 'c.', label='Test Metric')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'LossAccuracy.png')
		plt.close()
			
	def print_info(self, epoch):
		logger.info('[Test]  Epoch: %i' % epoch)
		logger.info('[Test]  loss: %.4f, metric: %.4f' % (self.test_loss, self.test_metric))
