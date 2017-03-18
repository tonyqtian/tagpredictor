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

pdtraining = '../data/cooking.csv'
pdtest = '../data/biology.csv'
output_dir = '../output'

logger = logging.getLogger(__name__)
mkdir(output_dir)
timestr = setLogger(out_dir=output_dir)

# process train and test data
_, train_title, train_content, train_tag = get_pdTable(pdtraining)
_, test_title, test_content, test_tag = get_pdTable(pdtest)

train_body = tableMerge([train_title, train_content])
test_body = tableMerge([test_title, test_content])

train_body, train_maxLen = tokenizeIt(train_body, clean=True)
test_body, test_maxLen = tokenizeIt(test_body, clean=True)
train_tag, _ = tokenizeIt(train_tag)
test_tag, _ = tokenizeIt(test_tag)
maxInputLength = max(train_maxLen, test_maxLen)
outputLength = 5

#  embedding: glove-pre-train(train_x, test_x)
# embd = makeGlove(vocab, train_x, test_x)
embdw2v, vocabDict, vocabReverseDict = makeEmbedding([train_body, test_body])
# vocabDict, vocabReverseDict, maxInputLength = createVocab([train_body, test_body])
# logger.info(vocabDict)
# logger.info(vocabReverseDict)

# word to padded numerical np array
train_x = word2num(train_body, vocabDict, maxInputLength)
test_x = word2num(test_body, vocabDict, maxInputLength)
train_y = word2num(train_tag, vocabDict, outputLength)
train_y = to_categorical2D(train_y, len(vocabDict))
test_y = word2num(test_tag, vocabDict, outputLength)
test_y = to_categorical2D(test_y, len(vocabDict))

# create model 
rnnmodel = getModel(maxInputLength, outputLength, len(vocabDict), embd=embdw2v, embd_dim=100, rnn_opt='gpu')
rnnmodel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
rnnmodel.summary()

# train and test model
evl = Evaluator(output_dir, timestr, 'categorical_accuracy', test_x, test_y)
earlystop = EarlyStopping(patience = 5, verbose=1, mode='auto')
rnnmodel.fit(train_x, train_y, validation_split=0.2, batch_size=256, nb_epoch=20, callbacks=[earlystop, evl])
# rnnmodel.fit(train_x, train_y, validation_split=0.2, batch_size=8, nb_epoch=2)
