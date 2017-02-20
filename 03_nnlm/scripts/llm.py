import numpy as np

from collections import defaultdict

wids = defaultdict(lambda: len(wids))

# Writing a function to read in the training and test corpora, and converting the words into numerical IDs.
def read_corpus(file):
  # for each line in the file, split the words and turn them into IDs like this:
  print file
  f = open(file)
  for line in f:
    line = line.split('\n')[0]
    words = line.split()
    for word in words:
         wid = wids[word]

def print_words():
  for wid in wids:
      print wid, wids[wid]
