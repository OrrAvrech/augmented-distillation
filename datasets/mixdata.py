from pathlib import Path
import numpy as np
import os
import pandas as pd


class MIXDATA:
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

        root_path = Path(root)
        fname_csv = root_path / subname / f"{subname}.csv"

        train, val, mu, st, unid, UNK, inv_feature_levels_train, inv_feature_levels_val, header_names, vocabs, continuous_featnames = load_and_normalize(fname_csv, subname)

        self.train = self.Data(train)
        self.val = self.Data(val)
        # consider val as test too
        self.test = self.Data(val)

        self.n_dims = self.train.x.shape[1]
        self.mu = mu
        self.st = st
        self.unid = unid
        self.UNK = UNK
        self.inv_feature_levels_train = inv_feature_levels_train
        self.inv_feature_levels_val = inv_feature_levels_val
        self.header_names = header_names
        self.vocabs = vocabs
        self.continuous_featnames = continuous_featnames


def replace_missing_values(dataset_split, cont_featnames):
    '''
        replace missing values with mean
    '''
    for feat in dataset_split.columns:

        if feat in cont_featnames and dataset_split[feat].isnull().sum() >= 1:# means it is missing
            mx = dataset_split[feat].mean()
            dataset_split[feat].fillna(mx, inplace=True)


def get_uniform_numbers(featsname):
    '''
    return uniform numbers
    '''
    np.random.seed(100)
    uniform ={}
    for i in featsname:
        uniform[i] = np.random.rand()

    return uniform


def build_vocab(X, cont_featnames, unk=None):

    feat_categories = {}
    num_classes = {}

    for feature in X.columns:

        if feature not in cont_featnames:
            feature_vals = X[feature].copy()

            # replace missing values
            feature_vals.fillna(np.random.choice(feature_vals.dropna()), inplace =True)
            
            if feature in feat_categories:
                raise ValueError( "Same name for multiple columns ")
            # now save vocabs. remember unk should have zero as index
            feat_categories[feature] = sorted(list(feature_vals.unique()))
            num_classes[feature] = len(feat_categories[feature])

    return feat_categories, num_classes


def convert_data(X, vocabs, uniform_numbers=None, mean_columns=None, std_columns=None, unk=None, use_mst = True):

    feature_levels = {}
    inv_feature_levels = {}
    if mean_columns is None and std_columns is None and use_mst == False:
        mean_columns = {}
        std_columns  = {}

    for feature in X.columns:
        if feature in vocabs:
            feature_levels[feature] = {}
            inv_feature_levels[feature] = {}
            feature_vals = X[feature].copy()

            # replace missing values with UUnnKK
            feature_vals.fillna(np.random.choice(feature_vals.dropna()), inplace =True)


            #feat_categories = sorted(list(feature_vals.unique()))
            feat_categories = vocabs[feature]

            for j in range(len(feat_categories)):
                feat_category_j = feat_categories[j]
                feature_levels[feature][feat_category_j] = j
                inv_feature_levels[feature][j] = feat_category_j

            X.loc[:,feature] = pd.Series(feature_vals.map(feature_levels[feature]), index = X.index)

            ##################
            ## normalize data using dequantization 
            ##################
            X.loc[:,feature] = (X.loc[:,feature] + uniform_numbers[feature])/len(feat_categories)

        else:
            ##################
            ## normalize data
            ##################
            if use_mst == False:
                mean_columns[feature] = X.loc[:,feature].mean()
                std_columns[feature]  = X.loc[:,feature].std()

            X.loc[:,feature]  = (X.loc[:,feature] - mean_columns[feature])/(std_columns[feature] + 1e-7)   # [N, D]

    return inv_feature_levels, mean_columns, std_columns


def load_and_normalize(fname_csv):
    
    '''
        Load data and normalize
    '''

    ######
    # remove missing values if dataset has it
    ######
    df = pd.read_csv(fname_csv, header=0)
    header_names = [i for i in Xtrain.columns]

    if os.path.exists(fname_csv + "/numericalfeatures.csv") and os.path.getsize(fname_csv + "/numericalfeatures.csv") > 0:
        continuous_featnames = pd.read_csv(fname_csv + "/numericalfeatures.csv", header=None).values
        continuous_featnames = {str(i[0]) for i in continuous_featnames}

        # replace missing values
        replace_missing_values(Xtrain, continuous_featnames)
        replace_missing_values(Xval, continuous_featnames)

    else:
        continuous_featnames = {}

    # now map non-numerical feat to numerical one 
    UNK = 'UUnnKK'

    ######
    ### build vocabs and convert symbols to numbers 
    ######
    vocabs, num_classes = build_vocab(pd.concat([Xtrain, Xval]), cont_featnames=continuous_featnames, unk=UNK)

    # generate uniform number for dequantization
    unid = get_uniform_numbers(vocabs.keys())
    inv_feature_levels_train, mu, st = convert_data(Xtrain, vocabs, uniform_numbers=unid, unk=UNK, use_mst=False)
    inv_feature_levels_val, _, _ = convert_data(Xval, vocabs, uniform_numbers=unid, mean_columns=mu, std_columns=st, unk=UNK)

    return Xtrain.values, Xval.values, mu, st, unid, UNK, inv_feature_levels_train, inv_feature_levels_val, header_names, vocabs, continuous_featnames