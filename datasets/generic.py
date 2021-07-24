from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class GENERIC:
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

    def __init__(self, data_path):
        fname_csv = Path(data_path)
        features, train, val, mu, st, header_names = load_and_normalize(fname_csv)

        self.features = self.Data(features)
        self.train = self.Data(train)
        self.val = self.Data(val)
        # consider val as test too
        self.test = self.Data(val)

        self.n_dims = self.train.x.shape[1]
        self.mu = mu
        self.st = st
        self.header_names = header_names


def replace_missing_values(dataset_split):
    """
        replace missing values with mean
    """
    for idx, i in enumerate(dataset_split.isnull().sum()):

        if i == 1:  # means it is missing nan or null

            mx = dataset_split[str(idx)].mean()
            dataset_split[str(idx)].fillna(mx, inplace=True)


def load_and_normalize(fname_csv):
    """
        Load data and normalize
    """
    ######
    # remove missing values if dataset has it
    ######
    df = pd.read_csv(fname_csv, header=0)
    features = df[df.columns[:-1]].to_numpy()
    labels = df[df.columns[-1]].to_numpy()
    header_names = [i for i in df.columns]
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=1)

    #######
    # normalize data
    # [N, D]
    #######
    mu = np.mean(x_train, axis=0)   # [D]
    st = np.std(x_train, axis=0)    # [D]

    x_train = (x_train - mu)/(st + 1e-7)   # [N, D]
    x_val = (x_val - mu)/(st + 1e-7)    # [N, D]

    return features, x_train, x_val, mu, st, header_names
