
from itertools import tee, islice
from collections import Counter
import  re, math
import numpy as np
from math import exp

file_name = '../data/en-de/train.en-de.en'
test_file = '../data/en-de/test.en-de.en'
test_file = 'test.test'
#file_name = 'test'

alpha_1 = 0.245
alpha_unk = 0.735
alpha_2 = 0.02

# String and length of ngram needed
def store_ngrams(s, n):
  tlst = s
  while True:
    a, b = tee(tlst)    
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break
    
def eqn8(strng):
  l = len(strng.split())
  if l == 1:
    try: 
       #print k[strng], sum(k.values())
       estimate = (1 - alpha_1) * (float(k[strng]) / sum(k.values())) + alpha_unk * exp(1e-7)
    except KeyError:
      estimate =  alpha_unk * exp(1e-7)
    return estimate
  else:
    c = strng
    strng = strng.split()[:-1]
    p = ' '.join(tk for tk in strng)
    try: 
      #print k[c], k[p]
      estimate = (1 - alpha_2) * (float(k[c]) / k[p]) + (1 - alpha_1) * (float(k[p]) / sum(k.values()))  + alpha_unk * exp(1e-7)
    #return float(k[c]) / k[p]
    except KeyError:
      estimate =  alpha_unk * exp(1e-7)
    return estimate
 
def ngrams_onlyn(lst, n):
  lst = lst.split(' ')
  output = {}
  for i in range(len(lst)-n+1):
    g = ' '.join(lst[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output 
    
def ngrams(lst, n):
  lst = lst.split(' ')
  output = {}
  while n > 0:
   for i in range(len(lst)-n+1):
    g = ' '.join(lst[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
   n = n -1 
  return output
    
n = 2
k = {}    
f = open(file_name)
for line in f:
  line = '<s> ' + line.split('\n')[0] + ' </s>'
  #print "The line is " + str(line)
  t =  ngrams(line,n)
  #print t
  for ke in t:
    #print ke
    if k.has_key(ke):
       k[ke] = int(k[ke]) + int(t[ke])
    else:
       k[ke] = int(t[ke])



strng = "you know"
mle = eqn8(strng)
print mle  
ppl = exp(1-(math.log(mle)/len(strng.split())))
print ppl
  
ppl_array = []  
f = open(test_file)
for line in f:
  line = line.split('\n')[0] + ' </s>'
  print line
  t =  ngrams_onlyn(line,n)
  for ke in t:
   if ke in k: 
    mle = eqn8(ke)
    length_string = 2
    s = ke
   else:
     if ke.split()[-1] in k:
       mle = eqn8(ke.split()[-1])
       length_string = 1
       s = ke.split()[-1]
     else:
       mle = alpha_unk * exp(1e-7)
       length_string = 1
       s = ke.split()[-1]
   ppl = exp(-(math.log(mle)/length_string))
   print s,ppl
   ppl_array.append(ppl)

print np.mean(ppl_array)    
    
  
  
