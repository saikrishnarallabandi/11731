from seq2seq_beam import Attention_Batch as AB

ab = AB(src_fname, test_src_fname, tgt_fname, test_tgt_fname)
ab.initiate_params()
ab.train_generate_model()
ab.test_only()
