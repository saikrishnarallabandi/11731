
from itertools import tee, islice
from collections import Counter
import  re, math, os
import numpy as np
from math import exp

file_name = '../data/en-de/train.en-de.tok.en'

alpha_unk = 0.02
alpha_1 = 0.245
alpha_2 = 0.735

def eqn8(strng):
  l = len(strng.split())
  if l == 1:
    estimate = (1 - alpha_1) * (float(unigram_dict[strng]) / sum(unigram_dict.values())) + alpha_unk * exp(1e-7)
    return estimate
  else:
    c = strng
    strng = strng.split()[:-1]
    p = ' '.join(tk for tk in strng)
    estimate = (1 - alpha_2) * (float(bigram_dict[c]) / unigram_dict[p]) + (1 - alpha_1) * (float(unigram_dict[p]) / sum(unigram_dict.values()))  + alpha_unk * exp(1e-7)
    return estimate
  
def corpus_perplexity(arr):  
  a = sum(np.log2(arr))
  b = -1.0 
  c = len(arr)
  print np.power(2, (a * b / c))
  #corpus_perplexity_v2(arr)

def corpus_perplexity_v3(arr, l):  
  a = sum(np.log2(arr))
  b = -1.0 
  c = l
  print "Perplexity: " , np.power(2, (a * b / c))
  
def corpus_perplexity_v2(arr):
  p = 1.0
  for a in arr:
    p = p * a
  lp = np.log2(p)
  print np.exp(-1.0 * lp / len(arr))

  
def combine_dicts(base_dict, small_dict):
     for key in small_dict:
         if base_dict.has_key(key):
	   base_dict[key] = int(base_dict[key]) + int(small_dict[key])
	 else:
	   base_dict[key] = int(small_dict[key])
     return base_dict 
   
def remove_singletons(base_dict):
     for key in base_dict:
        #print key, base_dict[key]
        if base_dict[key] < 2:
	  #print key
	  base_dict[key] = 0

def get_ngrams(lst, n):
      lst = lst.split()
      output = {}
      for i in range(len(lst) -n + 1):
	    g = ' '.join(lst[i:i+n])
            output.setdefault(g, 0)
            output[g] += 1
      return output

n = 2
bigram_dict = {}
unigram_dict = {}

f = open(file_name)
sentence_count = 0
for line in f:
  sentence_count += 1
  line = '<s> ' + line.split('\n')[0] + ' </s>'
  #line =  line.split('\n')[0] 
  bigrams = get_ngrams(line,2)
  bigram_dict = combine_dicts(bigram_dict,bigrams)
  unigrams = get_ngrams(line,1)
  unigram_dict = combine_dicts(unigram_dict,unigrams)

#print "Populated bigram and unigrams"
print "Total Sentences: ", sentence_count
print "Total words: ", sum(unigram_dict.values())
unigram_dict_original = unigram_dict
remove_singletons(unigram_dict)
print "Removed unigram singletons"
print "Total words: ", sum(unigram_dict.values())
#bigram_dict_original = bigram_dict
#remove_singletons(bigram_dict)
#print "Removed bigram singletons"
