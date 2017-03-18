'''
Created on Mar 17, 2017

@author: tonyq
'''
import pandas as pd
import re
# from tqdm._tqdm import tqdm
from bs4 import BeautifulSoup
from tqdm._tqdm import tqdm

def get_words(text):
	word_split = re.compile('[^a-zA-Z0-9_\\+\\-]')
	return [word.strip().lower() for word in word_split.split(text)]

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

cooking = pd.read_csv("../data/biology.csv")
totalen = len(cooking.tags)
print('Total size: ', totalen)
fulllist = list(zip(cooking.title, cooking.content, cooking.tags))
del cooking
tagLength = {}
tagset = set()
contentset = set()
wdFrq = {}
total_words = 0

for (title, content, tag) in tqdm(fulllist):
	ws = tag.split(' ')
	tagset.update(ws)
	try:
		tagLength[len(ws)] += 1
	except KeyError:
		tagLength[len(ws)] = 1
		
	text = stripTagsAndUris(title)
	text = get_words(text)
	contentset.update(text)
	for wd in text:
		try:
			wdFrq[wd] += 1
		except KeyError:
			wdFrq[wd] = 1
		total_words += 1
	
	text = stripTagsAndUris(content)
	text = get_words(text)
	contentset.update(text)
	for wd in text:
		try:
			wdFrq[wd] += 1
		except KeyError:
			wdFrq[wd] = 1
		total_words += 1
		
print('Sorting Frq Word Dict...')
import operator
sorted_word_freqs = sorted(wdFrq.items(), key=operator.itemgetter(1), reverse=True)
vocab_size = 0
frqContentset = set()
for _, freq in sorted_word_freqs:
	if freq > 1:
		vocab_size += 1
for word, _ in sorted_word_freqs[:vocab_size]:
	frqContentset.add(word)

print('Static...')
tagHit = {}
tagPartial = {}
frqTagHit = {}
frqTagPartial = {}
for (title, content, tag) in fulllist:
	ws = tag.split(' ')
	tagset.update(ws)
	hitset = contentset.intersection(set(ws))
	try:
		tagPartial[len(hitset)] += 1
	except KeyError:
		tagPartial[len(hitset)] = 1
		
	if len(hitset) == len(ws):
		updateLength = len(hitset)
	else:
		updateLength = 0
	try:
		tagHit[updateLength] += 1
	except KeyError:
		tagHit[updateLength] = 1

	frqHitset = frqContentset.intersection(set(ws))
	try:
		frqTagPartial[len(frqHitset)] += 1
	except KeyError:
		frqTagPartial[len(frqHitset)] = 1
		
	if len(frqHitset) == len(ws):
		updateLength = len(frqHitset)
	else:
		updateLength = 0
	try:
		frqTagHit[updateLength] += 1
	except KeyError:
		frqTagHit[updateLength] = 1
		
print(tagset)
print('Tag size: ', len(tagset))
print('Tag size distribution')
print(tagLength)
print('Partial hit distribution')
print(tagPartial)
print('Partial hit rate: ', (totalen - tagPartial[0])/totalen)
print('Exact hit distribution')
print(tagHit)
print('Max hit rate: ', (totalen - tagHit[0])/totalen)

print('Frq Partial hit distribution')
print(frqTagPartial)
print('Partial hit rate: ', (totalen - frqTagPartial[0])/totalen)
print('Frq Exact hit distribution')
print(frqTagHit)
print('Max hit rate: ', (totalen - frqTagHit[0])/totalen)

print('Content vocab: ', len(contentset))
interset = tagset.intersection(contentset)
print(interset)
print('Interset size: ', len(interset))
print('Hit rate: ', len(interset)/len(tagset))

print('Content vocab: ', len(frqContentset))
interset = tagset.intersection(frqContentset)
print(interset)
print('Frq Interset size: ', len(interset))
print('Hit rate: ', len(interset)/len(tagset))