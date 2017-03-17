'''
Created on Mar 16, 2017

@author: tonyq
'''
from xml.dom import minidom
from tqdm._tqdm import tqdm
import csv

with open("../data/physics_sample.csv", "w", encoding='utf8') as wfh:
	writer = csv.writer(wfh)
	writer.writerow(['id','title','content','tags'])
	print('Processing XML...')
	xmldoc = minidom.parse('../data/tail_sample.xml')
	print('Getting content...')
	itemlist = xmldoc.getElementsByTagName('row')
	print(len(itemlist))
	print(itemlist[0].attributes['Id'].value)
	print('Writing to file...')
	for s in tqdm(itemlist):
		try:
			ids = s.attributes['Id'].value
			tags = s.attributes['Tags'].value
			body = s.attributes['Body'].value
			title = s.attributes['Title'].value
		except KeyError:
# 			print('Unparsed line: ', s.attributes['Id'].value)
			pass
		tags = tags.replace('<','')
		tags = tags.replace('>',' ')
		writer.writerow([ids,title,body,tags.strip(' ')])
	print('Finished')	