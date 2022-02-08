# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:20:27 2022

Code for Evaluating Performance of Minimum Edit Distance for Automatic
Spelling Correction for the course COMP-8730 for Winter 2022

@author: piranir
"""
# Step to download the wordnet corpus manually if not already done
#import nltk
#nltk.download() # Select 'wordnet' from the 'Corpora' tab and download

import datetime
import multiprocessing as mp
import a1nlp_util
import pickle
import numpy as np
import pandas as pd

SPELLING_ERROR_CORPUS_FILE = 'missp.dat'

from nltk.corpus import wordnet as wn

# Read the words from the WordNet dictionary D
dictionary = list(wn.all_lemma_names())

# Remove phrases that are marked as words (like "bird's_eye")
special_characters = """'"!@#$%^&*()+?=,<>/"""
dictionary = [word for word in dictionary 
              if not any(letter in special_characters for letter in word)]

dictionary = dictionary[:50000] # Selecting top 50,000 words

# Reading the Birkbeck spelling error corpus
with open(SPELLING_ERROR_CORPUS_FILE, 'r') as f:
    lines = f.read().splitlines()

lines = [word.lower() for word in lines]
lines = lines[:5000] # Selecting top 5,000 words
lines_processed = list()
spelling_error_corpus = dict()

i, j = 0, 0
while j < len(lines):
    if lines[i][0] == '$':
        j += 1
        while j < len(lines) and lines[j][0] != '$':
            j += 1
        lines_processed.append(lines[i:j])
    i = j

for line in lines_processed:
    spelling_error_corpus[line[0][1:]] = line[1:]

# Multiprocessing by available CPU cores for parallel processing
pool = mp.Pool(mp.cpu_count())

start_time = datetime.datetime.now()

# Computing the minimum edit distance between misspelled words in C
# and every word
spelling_error_word_med = pool.apply(a1nlp_util.calc_spelling_error_word_med, 
                                     args=(dictionary, spelling_error_corpus))

end_time = datetime.datetime.now()
print("Time taken: " + str(end_time - start_time))

# Storing a pickle file of the processed data with MED for later use
pickle_file = open('spelling_error_word_med.pkl', 'ab')
pickle.dump(spelling_error_word_med, pickle_file)
pickle_file.close()

# Converting processed data into a pandas DataFrame
spelling_error_word_med_df = pd.DataFrame(spelling_error_word_med,
                                           columns = ['spelling', 'error',
                                                      'word', 'med'])

# Selecting the top 10 spellings for every word
top_ten_grouped = spelling_error_word_med_df\
    .groupby(['spelling', 'error'])\
    .apply(lambda x: x.nsmallest(10, ['med']))\
    .reset_index(drop=True)

# Calculating success, i.e., if a dictionary word matches the original spelling
top_ten_grouped['success'] = np.where(
    top_ten_grouped['spelling'] == top_ten_grouped['word'], 1, 0)

# Computing the success at k measure s@k
top_ten_grouped['success_at_k'] = top_ten_grouped.groupby(
    ['spelling', 'error'])['success'].transform(pd.Series.cumsum)

# Selecting s@k for k = {1, 5, 10}
s_at_k = top_ten_grouped.groupby(['spelling', 'error'])\
    .nth([0,4,9])['success_at_k']

# Computing the overall average and averages for s@k for k = {1, 5, 10}
avg_s_at_k = s_at_k.groupby(['spelling', 'error']).transform(np.mean)

avg_s_at_k = avg_s_at_k.reset_index().drop_duplicates(ignore_index=True)

avg_s_at_k.to_csv('avg_s_at_k.csv', index=False)

overall_avg = np.mean(avg_s_at_k['success_at_k'])

s_at_1 = top_ten_grouped.groupby(['spelling', 'error'])\
    .nth(0)['success_at_k']
avg_s_at_1 = np.mean(s_at_1)

s_at_5 = top_ten_grouped.groupby(['spelling', 'error'])\
    .nth(4)['success_at_k']
avg_s_at_5 = np.mean(s_at_5)

s_at_10 = top_ten_grouped.groupby(['spelling', 'error'])\
    .nth(9)['success_at_k']
avg_s_at_10 = np.mean(s_at_10)

# Summarizing the results
avg_results = pd.DataFrame(
    [(avg_s_at_1, avg_s_at_5, avg_s_at_10, overall_avg)],
    columns=['Average s@1', 'Average s@5',
             'Average s@10', 'Overall Average'])
print(avg_results)