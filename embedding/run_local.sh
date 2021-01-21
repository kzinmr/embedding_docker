#!/bin/bash
window=32
min_count=10
dimension=256
max_vocab_size=50000
max_sentence_length=30000  # num tokens
python3 preprocess.py

w2v_model=/workspace/models/word2vec_sg_dim${dimension}_w${window}_mc${min_count}.model
python train_word2vec.py \
-i $wikipedia_wakati \
-o $w2v_model \
--min-count $min_count \
--max-vocab-size $max_vocab_size \
--max-sentence-length $max_sentence_length \
--window $window \
--dimension $dimension
