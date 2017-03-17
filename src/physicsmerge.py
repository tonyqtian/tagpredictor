'''
Created on Mar 17, 2017

@author: tonyq
'''
from tqdm._tqdm import tqdm
import pandas as pd
import csv

with open("../data/physics_full_sub.csv", "w", encoding='utf8') as wfh:
	with open("../data/physics_full_res.csv", "w", encoding='utf8') as wfh2:
		writer_sub = csv.writer(wfh)
		writer_sub.writerow(['id','title','content','tags'])
		writer_res = csv.writer(wfh2)
		writer_res.writerow(['id','title','content','tags'])
		phfull = pd.read_csv('../data/physics_full.csv')
		subm = pd.read_csv('../data/sample_submission_test.csv')
		phset = set(subm.id)
		phyfull = zip(phfull.id, phfull.title, phfull.content, phfull.tags)
		del subm
		del phfull
		for (id1, title, content, tag) in phyfull:
			if id1 in phset:
				writer_sub.writerow([id1,title, content, tag])
			else:
				writer_res.writerow([id1,title, content, tag])
			
# test = pd.read_csv("../data/physicsTags_subm.csv")
# test.to_csv("../data/physicsTags_subm_pd.csv", index=False)