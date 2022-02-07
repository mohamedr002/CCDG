# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset

from domainbed.utils.Dataset import DG_Dataset

DATASETS = [
    # Debug
    "DebugUniv",
    "DebugMultiv",
    # Time series
    "bearing"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 3001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 1            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override
    
    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class DebugUniv(Debug):
    INPUT_SHAPE = (128,)
    ENVIRONMENTS = ['0', '1', '2']

class DebugMultiv(Debug):
    INPUT_SHAPE = (3, 128)
    ENVIRONMENTS = ['0', '1', '2']

class MultipleEnvironmentTSFolder(MultipleDomainDataset):
    def __init__(self, root, dataset, test_envs, augment, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        environments = [f.name for f in os.scandir(root)]
        environments = sorted(environments)

        self.datasets = []
        for i, environment in enumerate(environments):

            path = os.path.join(root, environment)
            env_dataset = DG_Dataset(dataset=dataset, file_path=path)
            self.datasets.append(env_dataset)

        self.num_classes = len(self.datasets[-1].classes)

class bearing(MultipleEnvironmentTSFolder):
    INPUT_SHAPE = (4096,)
    ENVIRONMENTS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "bearing/")
        super().__init__(self.dir, 'bearing', test_envs, hparams['data_augmentation'], hparams)


class paderborn(MultipleEnvironmentTSFolder):
    INPUT_SHAPE = (5120,)
    ENVIRONMENTS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "paderborn/")
        super().__init__(self.dir, 'paderborn', test_envs, hparams['data_augmentation'], hparams)