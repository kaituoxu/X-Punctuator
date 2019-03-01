#!/usr/bin/env python

# Created on 2019/03
# Author: Kaituo XU (NPU-ASLP)

import argparse

import torch

from dataset import NoPuncTextDataset
from model import LstmPunctuator
from utils import add_punc_to_txt


parser = argparse.ArgumentParser(description="Punctuation prediction.")
parser.add_argument('--data', type=str, required=True, help='Text data to be punctuated')
parser.add_argument('--vocab', type=str, required=True, help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', type=str, required=True, help='Output punctuations vocab. (Don\'t include " ")')
parser.add_argument('--model_path', type=str, required=True, help='model path created by training')
parser.add_argument('--output', default='-', help='Write the punctuated text (default to stdout)')
parser.add_argument('--use_cuda', default=0, type=int)


def add_punctuation(args):
    # Load data
    dataset = NoPuncTextDataset(args.data, args.vocab, args.punc_vocab)

    # Load model
    model = LstmPunctuator.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Output function
    if args.output == "-":
        output, endline = print, ""
    else:
        ofile = open(args.output, "w", encoding='utf8')
        output, endline = ofile.write, "\n"

    # Add punctuation
    with torch.no_grad():
        for i, (input, txt_seq) in enumerate(dataset):
            # Prepare input
            input_lengths = torch.LongTensor([len(input)])
            input = input.unsqueeze(0)
            if args.use_cuda:
                input, input_lengths = input.cuda(), input_lengths.cuda()
            # Forward propagation
            scores = model(input, input_lengths)
            # Convert score to prediction result
            scores = scores.view(-1, scores.size(-1))
            _, predict = torch.max(scores, 1)
            predict = predict.data.cpu().numpy().tolist()
            # Add punctuation to text
            result = add_punc_to_txt(txt_seq, predict, dataset.class2punc)
            # Write punctuated text with to output
            output(result + endline)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    add_punctuation(args)
