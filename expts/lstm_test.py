import dynet as dy

model = dy.Model()
# M-> vocab, emb (Input)
M = model.add_lookup_parameters((10000,2))
# gen_R ->desired o/p size or  vocab, hidden
gen_R = model.add_parameters((2, 3))
# gen_bis -> vocab or desired o/p size
gen_bias = model.add_parameters((2,))
# Trainer
trainer = dy.SimpleSGDTrainer(model)

def enc(num, dim):
 if num % 2 == 1:
   dy.renew_cg()
   # ENC -> layer, inp_dim, hid_dim
   gen_R = model.add_parameters((2, dim))
   encoder = dy.LSTMBuilder(1,2,dim, model)

 else:
   dy.renew_cg()
   # ENC -> layer, inp_dim, hid_dim
   encoder = dy.LSTMBuilder(1,2,3, model)
   gen_R = model.add_parameters((2, dim)) 
 encoder_state = encoder.initial_state()
 state = encoder_state.add_input(M[num])
 y_t = state.output()
 R = dy.parameter(gen_R)
 bias = dy.parameter(gen_bias)
 r_t = bias + (R * y_t)
 err = dy.pickneglogsoftmax(r_t, num)
 print trainer.status()
 #err.backward()
 #trainer.update(0.01)
 return y_t, r_t, err

for k in range(10):
 print "   Epoch: ", k
 for v in range(0,10):
    state, output, err = enc(v, 3)
    print "Error Value: ", err.value()
    err.backward()
    trainer.update(0.1)
    #if v % 500 == 1:
    print v, err.value() #M[v].value(), state.value(), output.value(), err.value()
    #print '\n'
    #while err.value() > 0.8:
    #   state, output, err = enc(v, 10)
    #   print v, M[v].value(), state.value(), output.value(), err.value()
