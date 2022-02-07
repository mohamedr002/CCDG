'''
Helper functions to consolidate and visualize results
'''

import os
import sys
import json
import itertools
import torch
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib as mplt
from matplotlib import pyplot as plt
from torchvision.datasets.folder import default_loader
from PIL import Image



### get domain generalization performance

# performance on unseen domain, allows aggregating across unseen domains

def get_perf_gen(logs, unseen_index_list, best_model_type, num_seed=1):
    acc_dict = {i:[] for i in range(num_seed)} # keys are seeds
    for unseen_index in unseen_index_list:  
        logs_path = os.path.join(logs, str(unseen_index))

        f_prefix = 'brief_test_accuracy'
        files = sorted([os.path.join(logs_path,f) for f in os.listdir(logs_path) \
            if os.path.isfile(os.path.join(logs_path,f)) and f.startswith(f_prefix)])[:num_seed]        

        # iterate through results of different seeds
        # select result for target domain under best_model_type
        for i,f in enumerate(files):
            df = pd.read_csv(f)
            df = df[df.best_model_type == best_model_type]
            df = df[df.domain == unseen_index]
            assert df.shape[0] == 1 
            df_acc = df.overall_acc.values[0]
            acc_dict[i].append(df_acc)
    
    # aggregate across domains per seed
    acc_agg = [np.mean(acc_dict[i]) for i in range(num_seed)]
    print('Mean:', np.mean(acc_agg))
    print('Standard deviation:', np.std(acc_agg))
    return(acc_agg)
    