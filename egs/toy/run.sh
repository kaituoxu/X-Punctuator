#!/bin/bash

. ./path.sh

train.py --train_data data/train20 --valid_data data/train20 --vocab data/vocab --punc_vocab data/punc_vocab --num_embeddings 5002 --batch_size 5 --print_freq 1 --use_cuda 0

add_punctuation.py --data data/train20_nopunc --vocab data/vocab --punc_vocab data/punc_vocab --model_path exp/temp/final.pth.tar --output result
