#!/usr/bin/env python

# Created on 2019/03
# Author: Kaituo XU (NPU-ASLP)

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from dataset import PuncDataset
from model import LstmPunctuator

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser(description="Punctuation prediction analyzer.")
parser.add_argument('--data', type=str, required=True, help='Text data to be analyzed.')
parser.add_argument('--vocab', type=str, required=True, help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', type=str, required=True, help='Output punctuations vocab. (Don\'t include " ")')
parser.add_argument('--model_path', type=str, required=True, help='model path created by training')
parser.add_argument('--use_cuda', default=0, type=int)


def analyze(args):
    # Load data
    dataset = PuncDataset(args.data, args.vocab, args.punc_vocab)

    # Load model
    model = LstmPunctuator.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    labels = np.array([])
    predicts = np.array([])

    for i, (word_id_seq, label_id_seq) in enumerate(dataset):
        # Prepare input
        input_lengths = torch.LongTensor([len(word_id_seq)])
        input = word_id_seq.unsqueeze(0)
        if args.use_cuda:
            input, input_lengths = input.cuda(), input_lengths.cuda()
        # Forward propagation
        scores = model(input, input_lengths)
        # Convert score to prediction result
        scores = scores.view(-1, scores.size(-1))
        _, predict = torch.max(scores, 1)
        predict = predict.data.cpu().numpy()
        # accumulate
        assert(len(label_id_seq) == len(predict))
        labels = np.append(labels, label_id_seq)
        predicts = np.append(predicts, predict)

    assert(len(labels) == len(predicts))

    # For each punctuation
    precision, recall, fscore, support = score(labels, predicts)
    accuracy = accuracy_score(labels, predicts)

    print("Multi-class accuracy: %.2f" % accuracy)
    FORMAT = "{}|{:^12}|{:^12}|{:^12}"
    SPLIT = "-"*(12*4+3)
    print(SPLIT)
    print(FORMAT.format("Punctuation ", "Precision", "Recall", "F-Score"))
    print(SPLIT)
    f = lambda x : round(x, 2)
    for (k, v) in sorted(dataset.punc2id.items(), key=lambda x:x[1]):
        if v >= len(precision): continue
        if k == " ":
            k = "  "
        print(FORMAT.format(" "*5+k+" "*5, f(precision[v]), f(recall[v]), f(fscore[v])))
    print(SPLIT)

    # For punc and non-punc, 0 is non-punc, 1 is punc
    labels = labels > 0
    predicts = predicts > 0
    precision, recall, fscore, support = score(labels, predicts, pos_label=1, average='binary')
    accuracy = accuracy_score(labels, predicts)
    print(FORMAT.format("punc/nonpunc", f(precision), f(recall), f(fscore)))
    print(SPLIT)
    print("Binary-class accuracy: %.2f" % accuracy)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    analyze(args)
