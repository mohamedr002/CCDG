# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from collections import Counter

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            network = network.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def accuracy_per_class(network, loader, weights, device, num_classes):
    correct_per_class = [0]*num_classes
    total_per_class = [0]*num_classes
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            network = network.to(device)
            p = network.predict(x)
            if p.size(1) == 1:
                for i in range(num_classes):
                    y_i = torch.tensor([yj for j, yj in enumerate(y) if y[j] == i]).to(device)
                    if len(y_i) > 0:
                        p_i = torch.stack([pj for j, pj in enumerate(p) if y[j] == i], dim = 0)
                        correct_per_class[i] += (p_i.gt(0).eq(y_i).float()).sum().item()
            else:
                for i in range(num_classes):
                    y_i = torch.tensor([yj for j, yj in enumerate(y) if y[j] == i]).to(device)
                    if len(y_i) > 0:
                        p_i = torch.stack([pj for j, pj in enumerate(p) if y[j] == i], dim = 0)
                        correct_per_class[i] += (p_i.argmax(1).eq(y_i).float()).sum().item()
            for i in range(num_classes):
                y_i = [yj for j, yj in enumerate(y) if y[j] == i]
                total_per_class[i] += len(y_i)
    network.train()

    acc_per_class = [correct_per_class[i]/total_per_class[i] for i in range(num_classes)]

    return acc_per_class, total_per_class

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def domain_contrastive_loss(domains_features, domains_labels, temperature,device):
    # masking for the corresponding class labels.
    anchor_feature = domains_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    labels = domains_labels
    labels= labels.contiguous().view(-1, 1)
    # Generate masking for positive and negative pairs.
    mask = torch.eq(labels, labels.T).float().to(device)
    # Applying contrastive via product among the features
    # Measure the similarity between all samples in the batch
    # reiterate fact from Linear Algebra if u and v two vectors are normalised implies cosine similarity
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

    # for numerical stability
    # substract max value from the output
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # create inverted identity matrix with same shape as mask.
    # only diagnoal is zeros and all others are ones
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device),
                                0)
    # mask-out self-contrast cases
    # the diagnoal represent same samples similarity i=j>> we need to remove this
    # remove the true labels mask
    # all ones in mask would be only samples with same labels
    mask = mask * logits_mask

    # compute log_prob and remove the diagnal
    # remove same features from the normalized contrastive matrix
    # The denominoator of the equation samples
    exp_logits = torch.exp(logits) * logits_mask

    # substract the whole multiplications from the negative pair and positive pairs
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mask_sum = mask.sum(1)
    zeros_idx = torch.where(mask_sum == 0)[0]
    mask_sum[zeros_idx] = 1

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    # loss
    loss = (- 1 * mean_log_prob_pos)
    loss = loss.mean()

    return loss
