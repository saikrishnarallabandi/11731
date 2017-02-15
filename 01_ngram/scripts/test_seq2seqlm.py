from seq2seq import ngramlm as NLM
import numpy as np

lm_01 = NLM(2)
lm_01.read_corpus('../data/en-de/train.en-de.low.en')
#lm_01.print_dicts()
#lm_01.save_dicts()
#t = lm_01.eqn8("hello")
#print "hello ", lm_01.get_sentence_perplexity("hello", 0)
#print '\n'
#print "hello this is cool \n", lm_01.get_sentence_perplexity("hello this is cool",0)

#print np.power(2, (-1.0 * np.log2(t)))
#lm_01.get_file_perplexity('../data/en-de/test.en-de.low.en')
lm_01.get_file_perplexity('../data/en-de/test.en-de.low.en')