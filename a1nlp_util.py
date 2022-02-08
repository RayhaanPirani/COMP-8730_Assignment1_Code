# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 22:51:59 2022

Python library containing the utility functions for Assignment 1 for
the course COMP-8730 for Winter 2022

@author: piranir
"""

# Computes and returns the minimum edit distance required to convert
# str1 to str2 using the dynamic programming approach
def min_edit_dist(str1, str2):
    if(len(str1) > len(str2)):
        str1, str2 = str2, str1
    
    dist1 = range(len(str1) + 1)
    
    for i2, c2 in enumerate(str2):
        dist2 = [i2 + 1]
        for i1, c1 in enumerate(str1):
            if c1 == c2:
                dist2.append(dist1[i1])
            else:
                dist2.append(min(dist1[i1], dist1[i1 + 1], dist2[-1]) + 1)
        dist1 = dist2
    
    return dist1[-1]

# Computes the minimum edit distance for words in the corpus
# spelling_error_corpus using the specified dictionary and
# returns a list of tuples containing the spelling, error (misspelling),
# dictionary word, and the minimum edit distance between the misspelling and
# the dictionary word
def calc_spelling_error_word_med(dictionary, spelling_error_corpus):
    spelling_error_word_med = list()
    
    for spelling in spelling_error_corpus:
        errors = spelling_error_corpus[spelling]
        for error in errors:
            for word in dictionary:
                med = min_edit_dist(error, word)
                spelling_error_word_med_tuple = (spelling, str(error), word, med)
                spelling_error_word_med.append(spelling_error_word_med_tuple)
    
    return spelling_error_word_med