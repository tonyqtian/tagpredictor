'''
Created on Mar 21, 2017

@author: tonyq
'''
import numpy as np

def precision(y_true, y_pred, argm=False):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	assert len(y_true) == len(y_pred), 'incorrect input length: ' + str(len(y_true)) + ' and ' + str(len(y_pred))
	if argm:
		y_true = np.argmax(y_true, axis=-1)
		y_pred = np.argmax(y_pred, axis=-1)
	true_positive = 0
	predicted_positive = 0
	for (y_true_line, y_pred_line) in zip(y_true, y_pred):
		tmpIntersec = np.intersect1d(y_true_line, y_pred_line)
		if 0 in tmpIntersec:
			true_positive += len(tmpIntersec) - 1
		else:
			true_positive += len(tmpIntersec)
		predicted_positive += np.sum(np.around(np.clip(np.unique(y_pred_line), 0, 1)))
	precision = true_positive / (predicted_positive + np.finfo(float).eps)
	return precision

def recall(y_true, y_pred, argm=False):
	"""Recall metric.
	Only computes a batch-wise average of recall.
	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	assert len(y_true) == len(y_pred), 'incorrect input length: ' + str(len(y_true)) + ' and ' + str(len(y_pred))
	if argm:
		y_true = np.argmax(y_true, axis=-1)
		y_pred = np.argmax(y_pred, axis=-1)
	true_positive = 0
	possible_positive = 0
	for (y_true_line, y_pred_line) in zip(y_true, y_pred):
		tmpIntersec = np.intersect1d(y_true_line, y_pred_line)
		if 0 in tmpIntersec:
			true_positive += len(tmpIntersec) - 1
		else:
			true_positive += len(tmpIntersec)
		possible_positive += np.sum(np.around(np.clip(np.unique(y_true_line), 0, 1)))
	recall = true_positive / (possible_positive + np.finfo(float).eps)
	return recall


def f1_score_prec_rec(y_true, y_pred, beta=1, argm=False):
	"""Computes the F score.
	The F score is the weighted harmonic mean of precision and recall.
	Here it is only computed as a batch-wise average, not globally.
	This is useful for multi-label classification, where input samples can be
	classified as sets of labels. By only using accuracy (precision) a model
	would achieve a perfect score by simply assigning every class to every
	input. In order to avoid this, a metric should penalize incorrect class
	assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
	computes this, as a weighted mean of the proportion of correct class
	assignments vs. the proportion of incorrect class assignments.
	With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
	correct classes becomes more important, and with beta > 1 the metric is
	instead weighted towards penalizing incorrect class assignments.
	"""
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	assert len(y_true) == len(y_pred), 'incorrect input length: ' + str(len(y_true)) + ' and ' + str(len(y_pred))
	if argm:
		y_true = np.argmax(y_true, axis=-1)
		y_pred = np.argmax(y_pred, axis=-1)
	true_positive = 0
	predicted_positive = 0
	possible_positive = 0
	epsilon = np.finfo(float).eps
	for (y_true_line, y_pred_line) in zip(y_true, y_pred):
		y_pred_line = np.unique(y_pred_line)
		y_true_line = np.unique(y_true_line)
		tmpIntersec = np.intersect1d(y_true_line, y_pred_line, assume_unique=True)
		if 0 in tmpIntersec:
			true_positive += len(tmpIntersec) - 1
		else:
			true_positive += len(tmpIntersec)
		predicted_positive += np.sum(np.around(np.clip(y_pred_line, 0, 1)))
		possible_positive += np.sum(np.around(np.clip(y_true_line, 0, 1)))
	precision = true_positive / (predicted_positive + epsilon)
	recall = true_positive / (possible_positive + epsilon)
	# If there are no true positives, fix the F score at 0 like sklearn.
	if possible_positive == 0:
		return precision, recall, 0
	bb = beta ** 2
	fbeta_score = (1 + bb) * (precision * recall) / (bb * precision + recall + epsilon)
	return precision, recall, fbeta_score

