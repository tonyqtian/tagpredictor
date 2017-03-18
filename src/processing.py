'''
Created on Mar 17, 2017

@author: tonyq
'''
import logging
from keras.callbacks import EarlyStopping
from utils import setLogger, mkdir
from data_processing import get_pdTable, tableMerge, tokenizeIt, createVocab, word2num
from model_processing import getModel

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

train_body = tokenizeIt(train_body, clean=True)
test_body = tokenizeIt(test_body, clean=True)
train_tag = tokenizeIt(train_tag)
test_tag = tokenizeIt(test_tag)

vocabDict, vocabReverseDict, maxInputLength = createVocab([train_body, test_body])
outputLength = 5
# logger.info(vocabDict)
# logger.info(vocabReverseDict)

# word to padded numerical np array
train_x = word2num(train_body, vocabDict, maxInputLength)
test_x = word2num(test_body, vocabDict, maxInputLength)
train_y = word2num(train_tag, vocabDict, outputLength)
test_y = word2num(test_tag, vocabDict, outputLength)

# create model 
#  embedding: glove-pre-train(train_x, test_x)
# embd = makeGlove(vocab, train_x, test_x)
rnnmodel = getModel(maxInputLength, outputLength, len(vocabDict))
rnnmodel.compile(loss='mse', optimizer='rmsprop')

# train and test model
# evl = Evaluator(out_dir, timestr, metric, test_x, test_y)
earlystop = EarlyStopping(patience = 5, verbose=1, mode='auto')
# rnnmodel.fit(train_x, train_y, validation_split=0.2, batch_size=256, nb_epoch=50, callbacks=[earlystop, evl])
rnnmodel.fit(train_x, train_y, validation_split=0.2, batch_size=8, nb_epoch=2)
