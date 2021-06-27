from __future__ import print_function
import numpy as np
import os
import pandas as pd
import pickle

class REGSDATA:
    """
    The AutoGluon dataset.
    """
    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]    

    def __init__(self, root, subname):

        fname_csv = os.path.join(root, subname,'fold_0')

        train, val, mu, st, header_names = load_and_normalize(fname_csv, subname)
        
        self.train = self.Data(train)
        self.val = self.Data(val)
        # consider val as test too
        self.test = self.Data(val)

        self.n_dims = self.train.x.shape[1]
        self.mu = mu
        self.st = st
        self.header_names = header_names

def replace_missing_values(dataset_split):
    '''
        replace missing values with mean
    '''
    for idx, i in enumerate(dataset_split.isnull().sum()):

        if i >= 1:# means it is missing nan or null

            mx = dataset_split[str(idx)].mean()
            dataset_split[str(idx)].fillna(mx, inplace=True)


def load_and_normalize(fname_csv, subname):
    
    '''
        Load data and normalize
    '''
    ######
    # remove missing values if dataset has it
    ######
    Xtrain_0 = pd.read_csv(fname_csv + "/Xtrain.csv", header=0)
    Xval_0   = pd.read_csv(fname_csv + "/Xval.csv", header=0)
    header_names = [i for i in Xtrain_0.columns]

    replace_missing_values(Xtrain_0)
    replace_missing_values(Xval_0)

    # make sure no missing values
    if Xtrain_0.isnull().values.any() or Xtrain_0.isna().values.any() or \
       Xval_0.isnull().values.any()  or Xval_0.isna().values.any():
        raise ValueError( 'There must be no missing values')

    # now pd --> np
    Xtrain = Xtrain_0.values
    Xval   = Xval_0.values

    #######
    # normalize data
    # [N, D]
    #######
    mu = Xtrain.mean(axis=0) # [D]
    st = Xtrain.std(axis=0)  # [D]

    Xtrain  = (Xtrain - mu)/(st + 1e-7)   # [N, D]
    Xval    = (Xval - mu)/(st + 1e-7)    # [N, D]

    return Xtrain, Xval, mu, st, header_names