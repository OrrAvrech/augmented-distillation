import torch
import datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_obj):

        # data size
        self.N = data_obj.N
        self.data = data_obj.x

    def __getitem__(self, idx):
        x = self.data[idx]
        return x

    def __len__(self):
        return self.N


def openml_checker(iname):
    '''
        List of openml and their info
    '''
    opml = {
        'Australian': {'train.shape': (496, 14),'val.shape': (125, 14),'small': True},
        'MiniBooNE': {'train.shape': (114557, 50),'val.shape': (2500, 50),'small': False},
        'Amazon_employee_access': {'train.shape': (26992, 9),'val.shape': (2500, 9),'small': False},
        'blood-transfusion': {'train.shape': (538, 4),'val.shape': (135, 4),'small': True},
        'higgs': {'train.shape': (85745, 28), 'val.shape': (2500, 28),'small': False},
        'jasmine': {'train.shape': (2185, 143),'val.shape': (500, 143),'small': True},
        'nomao': {'train.shape': (28518, 118),'val.shape': (2500, 118),'small': False},
        'numerai28.6': {'train.shape': (84188, 21),'val.shape': (2500, 21),'small': False},
        'sylvine': {'train.shape': (4111, 20), 'val.shape': (500, 20), 'small': True},
        'phoneme': {'train.shape': (4363, 5), 'val.shape': (500, 5), 'small': True},
        'Covertype': {'train.shape': (517680, 54),'val.shape': (5230, 54),'small': False},
        'Helena': {'train.shape': (56176, 27),'val.shape': (2500, 27),'small': False},
        'Jannis': {'train.shape': (72859, 54),'val.shape': (2500, 54),'small': False},
        'Volkert': {'train.shape': (49979, 147),'val.shape': (2500, 147),'small': False},
        'connect-4': {'train.shape': (58301, 42),'val.shape': (2500, 42),'small': False},
        'jungle_chess_2pcs_raw_endgame_complete': {'train.shape': (37837, 6),'val.shape': (2500, 6),'small': False},
        'mfeat-factors': {'train.shape': (1440, 216),'val.shape': (360, 216),'small': True},
        'segment': {'train.shape': (1663, 18), 'val.shape': (416, 18), 'small': True},
        'vehicle': {'train.shape': (608, 18), 'val.shape': (153, 18), 'small': True},

    }

    if iname in opml:
        return True, opml[iname]['small'], opml[iname]

    else:
        return False, False, None


def mixopml_checker(iname):
    '''
        List of mix openml and their info
    '''
    mx = {
        'adult': {'train.shape': (41457, 14),'val.shape': (2500, 14),'small': False},
        'credit-g': {'train.shape': (720, 20), 'val.shape': (180, 20), 'small': True},
        }

    if iname in mx:
        return True, mx[iname]['small'], mx[iname]

    else:
        return False, False, None


def regsopml_checker(iname):
    '''
        List of mix openml and their info
    '''
    mx = {
        'protein-tertiary-structure': {'train.shape': (38657, 9),'val.shape': (2500, 9),'small': False},
        'concrete': {'train.shape': (741, 8), 'val.shape': (186, 8), 'small': True},
        'energy': {'train.shape': (552, 8), 'val.shape': (139, 8), 'small': True},
        'yacht': {'train.shape': (221, 6), 'val.shape': (56, 6), 'small': True},
        'kin8nm': {'train.shape': (6635, 8), 'val.shape': (738, 8), 'small': True},
        'naval-propulsion-plant': {'train.shape': (9666, 14),'val.shape': (1075, 14),'small': True},
        'power-plant': {'train.shape': (7749, 4),'val.shape': (862, 4),'small': True},
        'bostonHousing': {'train.shape': (364, 13),'val.shape': (91, 13),'small': True},
        'wine-quality-red': {'train.shape': (1151, 11),'val.shape': (288, 11),'small': True}
        }

    if iname in mx:
        return True, mx[iname]['small'], mx[iname]

    else:
        return False, False, None


def get_ds_type(dataset_name, data_dir):
    if openml_checker(dataset_name)[0]:
        dataset = getattr(datasets, 'GENERIC')(data_dir, dataset_name)

    elif mixopml_checker(dataset_name)[0]:
        dataset = getattr(datasets, 'MIXDATA')(data_dir, dataset_name)

    elif regsopml_checker(dataset_name)[0]:
        dataset = getattr(datasets, 'REGSDATA')(data_dir, dataset_name)

    else:
        raise ValueError(dataset_name + " is not supported")
    return dataset
