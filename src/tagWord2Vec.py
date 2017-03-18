'''
Created on Mar 17, 2017

@author: tonyq
'''
import pandas as pd
import re
from tqdm._tqdm import tqdm
from bs4 import BeautifulSoup

# def clean_html(raw_html):
# 	cleanr = re.compile('<.*?>')
# 	cleantext = re.sub(cleanr, '', raw_html)
# 	return cleantext

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

vocab = set()
# with open('../data/googleWord2Vec_300M_vocablist.txt', 'r', encoding='utf8') as fhd:
# 	for line in tqdm(fhd):
# 		vocab.add(line.strip().split()[0])
with open('../data/googleWord2Vec_300M_vocablist.txt', 'r', encoding='utf8') as fhd:
	for line in tqdm(fhd):
		vocab.add(line.strip())
print(len(vocab))

cooking = pd.read_csv("../data/physics_full_sub.csv")
totalen = len(cooking.tags)
print('Total size: ', totalen)

fulllist = zip(cooking.title, cooking.content, cooking.tags)
del cooking
contentSet = set()
tagSet = set()
for (title, content, tag) in tqdm(fulllist):
# 	tag = tag.replace('-','_')
	ws = set(tag.split(' '))
	tagSet.update(ws)
	
	text = stripTagsAndUris(title)
	text = get_words(text)
	contentSet.update(text)
	
	text = stripTagsAndUris(content)
	text = get_words(text)
	contentSet.update(text)

hitWords = tagSet.intersection(vocab)
contentHit = contentSet.intersection(vocab)
	
print('Hit tags')
print(hitWords)
print('Hit rate ', len(hitWords)/len(tagSet))
print(tagSet.difference(vocab))

print('Hit content')
print(contentHit)
print('Hit rate ', len(contentHit)/len(contentSet))
print(contentSet.difference(vocab))