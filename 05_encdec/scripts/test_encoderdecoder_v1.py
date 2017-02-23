from seq2seq import EncoderDecoder as ed
from seq2seq import nnlm as LM
import dynet as dy
model = dy.Model()     
trainer = dy.SimpleSGDTrainer(model)
filename = '../data/en-de/test.en-de.low.en'
#filename = 'txt.done.data'

lm = LM()
train_dict,wids = lm.read_corpus(filename)
ED = ed(len(wids))
M = model.add_lookup_parameters((len(wids), 2))


def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
       # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out





sentences  = []
f = open(filename)
for  line in f:
   line = line.strip()
   sentences.append(line)
cum_loss = 0
words = 1
for epoch in range(10):
 sent = 0 
 for sentence in sentences:
  sent = sent  + 1 
  if len(sentence.split()) > 2:
     losses, num_words = ED.encode(sentence, wids)
     if sent % 100 == 1:
        print  "       ", sentence, losses.value(), cum_loss/ words 
     gen_losses = losses.vec_value()
     #print "Gen_losses: ", gen_losses
     loss = dy.sum_batches(losses)
     cum_loss += loss.value()
     words += num_words
     loss.backward()
     #print "Back propagated the loss"
     trainer.update(1)
     print(generate(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))




''' 
enc = dy.LSTMBuilder(1, 2,128,model)
#print "Got an encoder"
state = enc.initial_state()
#print "Added initial state"
#state.add_input(M[wids['help']])
#print "Added help"
state = enc.initial_state(losses)
#print "Initialized it to last stage"
state.add_input(M[wids['help']])
'''
