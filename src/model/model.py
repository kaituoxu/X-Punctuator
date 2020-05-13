# Created on 2019/02
# Author: Kaituo XU (NPU-ASLP)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmPunctuator(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hidden_size, num_layers, bidirectional,
                 num_class):
        super(LstmPunctuator, self).__init__()
        # Hyper-parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        # Components
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=bool(bidirectional))
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_dim, num_class)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T
            input_lengths: N
        Returns:
            score: N x T x C
        """
        # Embedding Layer
        padded_input = self.embedding(padded_input)  # N x T x D
        # LSTM Layers
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        # Output Layer
        score = self.fc(output)
        return score

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['num_embeddings'], package['embedding_dim'],
                    package['hidden_size'], package['num_layers'],
                    package['bidirectional'], package['num_class'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'num_embeddings': model.num_embeddings,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'bidirectional': model.bidirectional,
            'num_class': model.num_class,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == '__main__':
    torch.manual_seed(123)
    V, D, H, L, B, C = 10, 3, 4, 2, True, 5
    N, T = 3, 6

    punctuator = LstmPunctuator(V, D, H, L, B, C)
    print(punctuator)

    padded_input = torch.randint(V, (N, T)).long()
    padded_target = torch.randint(C, (N, T)).long()
    padded_input[-1, -2:] = 0
    padded_target[-1, -2:] = -1
    input_lengths = torch.LongTensor([T, T, T-2])
    print(padded_input)
    print(padded_target)
    print(input_lengths)

    score = punctuator(padded_input, input_lengths)
    print(score)

    score = score.view(-1, score.size(-1))
    padded_target = padded_target.view(-1)
    loss = torch.nn.functional.cross_entropy(score, padded_target,
                                             ignore_index=-1,
                                             reduction='elementwise_mean')
    print(loss)
