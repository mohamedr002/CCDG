import os, sys
import re
import h5py
import numpy as np
import pickle as p
from scipy.io import loadmat
import torch
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, default_loader
from torch.utils.data import Dataset


# import ipdb

class DG_Dataset(Dataset):
    def __init__(self, dataset, file_path):

        self.configuration(dataset, file_path)
        self.load_data()

    def configuration(self, dataset, file_path):
        self.dataset = dataset
        self.file_path = file_path

    def load_data(self):
        if self.dataset == 'bearing':
            bearing = torch.load(self.file_path)
            samples = np.array(bearing['samples'])
            labels = np.array(bearing['labels'])
        elif self.dataset == 'paderborn':
            paderborn = torch.load(self.file_path)
            samples = np.array(paderborn['samples'])
            labels = np.array(paderborn['labels'])

        self.samples = samples
        self.labels = labels
        # shift the labels to start from 0
        self.labels -= np.min(self.labels)
        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.dataset == 'bearing':
            sample = self.samples[index]
        if self.dataset == 'paderborn':
            sample = self.samples[index]

        label = self.labels[index]
        output = [sample, label]

        return(tuple(output))

    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx