# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np

ALGORITHMS_TUNE = [
    'IRM',
    'GroupDRO',
    'Mixup',
    #'MLDG',
    'CORAL',
    'MMD',
    'DANN', 
    'CDANN', 
    'MTL', 
    'SagNet',
    'VREx',
    'RSC'
]

def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}
    
    hparams['data_augmentation'] = (True, True)
    hparams['class_balanced'] = (False, False)

    if dataset == 'bearing':
        hparams['cnn_hidden_dim'] = (32, 32)
        hparams['last_layer_dim'] = (976, 976)
    elif dataset == 'HHAR':
        hparams['cnn_hidden_dim'] = (128, 128)
    elif dataset == 'paderborn':
        hparams['cnn_hidden_dim'] = (32, 32)
        hparams['last_layer_dim'] = (1232, 1232)


    # follows configurations in DFDG
    hparams['lr'] = (1e-3, 1e-3)
    hparams['batch_size'] = (32, 32) # per domain
    hparams['step_size'] = (0.8, 0.8)
    hparams['weight_decay'] = (5e-5, 5e-5)
    hparams['erm_wt'] = (0.9, 0.9)
    hparams['cont_wt'] = (0.1, 0.1)

    if algorithm in ['DANN', 'CDANN']:
        hparams['lr_g'] = (5e-5, 10**random_state.uniform(-5, -3.5))
        hparams['lr_d'] = (5e-5, 10**random_state.uniform(-5, -3.5))

        hparams['lambda'] = (1.0, 10**random_state.uniform(-2, 2))
        hparams['weight_decay_g'] = (0., 10**random_state.uniform(-6, -2))
        hparams['weight_decay_d'] = (0., 10**random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2**random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0., 10**random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    elif algorithm == "RSC":
        hparams['rsc_f_drop_factor'] = (1/3, random_state.uniform(0, 0.5))
        hparams['rsc_b_drop_factor'] = (1/3, random_state.uniform(0, 0.5))
    elif algorithm == "SagNet":
        hparams['sag_w_adv'] = (0.1, 10**random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams['irm_lambda'] = (1e2, 10**random_state.uniform(-1, 5))
        hparams['irm_penalty_anneal_iters'] = (500, int(10**random_state.uniform(0, 4)))
    elif algorithm == "Mixup":
        hparams['mixup_alpha'] = (0.2, 10**random_state.uniform(-1, 1))
    elif algorithm == "GroupDRO":
        hparams['groupdro_eta'] = (1e-2, 10**random_state.uniform(-3, -1))
    elif algorithm == "MMD" or algorithm == "CORAL":
        hparams['mmd_gamma'] = (1., 10**random_state.uniform(-1, 1))
    elif algorithm == "MLDG":
        hparams['mldg_beta'] = (1., 10**random_state.uniform(-1, 1))
    elif algorithm == "MTL":
        hparams['mtl_ema'] = (.99, random_state.choice([0.5, 0.9, 0.99, 1.]))
    elif algorithm == "VREx":
        hparams['vrex_lambda'] = (1e1, 10**random_state.uniform(-1, 5))
        hparams['vrex_penalty_anneal_iters'] = (500, int(10**random_state.uniform(0, 4)))

    return hparams

def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, dummy_random_state).items()}

def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, random_state).items()}
