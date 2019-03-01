#!/usr/bin/env python

# Created on 2019/02
# Author: Kaituo XU (NPU-ASLP)

import argparse

import torch

from dataset import build_data_loader
from model import LstmPunctuator
from solver import Solver
from utils import IGNORE_ID, num_param


parser = argparse.ArgumentParser("X-Punctuator training")
# Data related
parser.add_argument('--train_data', type=str, required=True, help='Training text data path.')
parser.add_argument('--valid_data', type=str, required=True, help='Cross validation text data path.')
parser.add_argument('--vocab', type=str, required=True, help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', type=str, required=True, help='Output punctuations vocab. (Don\'t include " ")')
# Model hyper parameters
parser.add_argument('--num_embeddings', default=100000+2, type=int, help='Input vocab size. (Include <UNK> and <END>)')
parser.add_argument('--embedding_dim', default=256, type=int, help='Input embedding dim.')
parser.add_argument('--hidden_size', default=512, type=int, help='LSTM hidden size of each direction.')
parser.add_argument('--num_layers', default=2, type=int, help='Number of LSTM layers')
parser.add_argument('--bidirectional', default=1, type=int, help='Whether use bidirectional LSTM')
parser.add_argument('--num_class', default=5, type=int, help='Number of output classes. (Include blank space " ")')
# minibatch
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2 penalty)')
# Training config
parser.add_argument('--use_cuda', default=1, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--half_lr', default=0, type=int, help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', default=0, type=int, help='Early stop training when halving lr but still get small improvement')
parser.add_argument('--max_norm', default=5, type=float, help='Gradient norm threshold to clip')
# save and load model
parser.add_argument('--save_folder', default='exp/temp', help='Dir to save models')
parser.add_argument('--checkpoint', default=0, type=int, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar', help='model name')
# logging
parser.add_argument('--print_freq', default=10, type=int, help='Frequency of printing training infomation')
# visualizing loss using visdom
parser.add_argument('--visdom', type=int, default=0, help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', type=int, default=0, help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='X-Punctuator training', help='Identifier for visdom run')


def main(args):
    # Build data loader
    tr_loader = build_data_loader(args.train_data, args.vocab, args.punc_vocab,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.num_workers)
    cv_loader = build_data_loader(args.valid_data, args.vocab, args.punc_vocab,
                                  batch_size=args.batch_size, drop_last=False)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # Build model
    model = LstmPunctuator(args.num_embeddings, args.embedding_dim,
                           args.hidden_size, args.num_layers, args.bidirectional,
                           args.num_class)
    print(model)
    print("Number of parameters: %d" % num_param(model))
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # Build criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    # Build optimizer
    optimizier = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.l2)
    # Build Solver
    solver = Solver(data, model, criterion, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
