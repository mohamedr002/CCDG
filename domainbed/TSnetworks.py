# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import misc

# feature extractors 

class CNN_Deep(nn.Module):

    classifier_type = 'MLP_Deep'

    def __init__(self, in_channels, hparams):
        super(CNN_Deep, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = self.n_outputs = hparams['cnn_hidden_dim']
        self.out_layer_dim = hparams['last_layer_dim']

        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels, 8, kernel_size=64, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1),
            nn.Flatten(),
            nn.Linear(self.out_layer_dim, self.hidden_dim))

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0),self.in_channels,-1)
        features = self.encoder(src)
        return features 

class CNN_Shallow(nn.Module):

    classifier_type = 'MLP_Shallow'

    def __init__(self, in_channels, hparams):
        super(CNN_Shallow, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = self.n_outputs = hparams['cnn_hidden_dim']
        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels, 128, kernel_size=8, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256,128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.in_channels, -1)
        features = self.encoder(src)
        return features

# classifiers

class MLP_Deep(nn.Module):
    def __init__(self, hidden_dim, num_cls):
        super(MLP_Deep, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,  self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_cls))

    def forward(self, features):
        outputs = self.Classifier(features)
        return outputs

class MLP_Shallow(nn.Module):
    def __init__(self, hidden_dim, num_cls):
        super(MLP_Shallow, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_cls))

    def forward(self, features):
        outputs = self.Classifier(features.squeeze())
        return outputs

# wrappers

def Featurizer(input_shape, hparams, original_input_shape=None):
    """Auto-select an appropriate featurizer for the given input shape."""
    original_input_shape = input_shape if original_input_shape is None else original_input_shape
    if len(original_input_shape) == 1:
        if len(input_shape) == 1:
            in_channels = 1
        else:
            in_channels = input_shape[0]
        return CNN_Deep(in_channels = in_channels, hparams=hparams)
    elif len(original_input_shape) == 2:
        return CNN_Shallow(in_channels=input_shape[0], hparams=hparams)
    else:
        raise NotImplementedError

def Classifier(classifier_type, hidden_dim, num_cls):
    """Auto-select an appropriate classifier for the given input shape."""
    if classifier_type == 'MLP_Deep':
        return MLP_Deep(hidden_dim, num_cls)
    elif classifier_type == 'MLP_Shallow':
        return MLP_Shallow(hidden_dim, num_cls)
    else:
        raise NotImplementedError

# other networks

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        if len(input_shape) == 1:
            self.in_channels = 1
        else:
            self.in_channels = input_shape[0]

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 5, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 5, padding=padding),
        )

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0),self.in_channels,-1)        
        return self.context_net(src)        