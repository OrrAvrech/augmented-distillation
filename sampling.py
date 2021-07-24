import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from models.transformer import BRT
from utils.data import Dataset, get_ds_type
import numpy as np
import pandas as pd


def sample_single_conditional(data, model, num_iter, mid=None):
    conditionals = []
    for i in range(num_iter):
        with torch.no_grad():
            if mid is None:
                mid = np.random.randint(data.shape[1])
            batch_conditionals = model.batch_feed_forward(data, mid=mid)
            conditionals += list(batch_conditionals.numpy())
    return conditionals


def sample_batch_conditionals(data_loader, model, num_iter, mid=None):
    conditionals = []
    for i in range(num_iter):
        for batch_idx, data in enumerate(data_loader):
            data_fetch = data.to(model.device)
            with torch.no_grad():
                if mid is None:
                    mid = np.random.randint(data_fetch.shape[1])
                batch_conditionals = model.batch_feed_forward(data_fetch, mid=mid)
                conditionals += list(batch_conditionals.numpy())
    return conditionals


def sample_gibbs(rounds, data_loader, model, num_iter):
    joints = []
    for i in range(num_iter):
        for batch_idx, data in enumerate(data_loader):
            data_fetch = data.to(model.device)
            batch_joints, _ = model.gibbs_sampler(rounds=rounds,
                                                  real_data=data_fetch,
                                                  dim_orders=range(data_fetch.shape[1]))
            joints += list(batch_joints.numpy())
    return joints


def denormalize(samples, mu, std):
    samples = (samples + mu) * (std + 1e-7)
    return samples


def main():
    dataset_name = 'bivariate_normals'
    data_path = Path('./datasets/synth/bivariate_normals/bivariate_normals.csv')
    batch_size = 1
    sampler = 'cond'
    gibbs_rounds = 1
    num_iter = 2000
    model_path = Path('ck/bivariate_normals_2021_07_11_0043/bivariate_brt_bivariate_normals_2021_07_11_0043.pt')
    model_params_path = Path('ck/bivariate_normals_2021_07_11_0043/bivariate_brt_bivariate_normals_2021_07_11_0043.json')
    aug_samples_csv_path = data_path.parent / f"{dataset_name}_{sampler}_{num_iter}.csv"

    dataset = get_ds_type(dataset_name, data_path)
    train_ds = Dataset(data_obj=dataset.train)
    data_loader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=False)

    with open(model_params_path, 'r') as f:
        params = json.load(f)['args']
    model = BRT(hidden_size=params['hidden_size'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                n_components=params['n_components'],
                device=torch.device('cpu'))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if sampler == 'gibbs':
        dists = sample_gibbs(gibbs_rounds, data_loader, model, num_iter)
    else:
        # dists = sample_batch_conditionals(data_loader, model, num_iter)
        dists = sample_single_conditional(torch.tensor([[-0.365398493, -4.203630746]]), model, num_iter, mid=0)

    samples = denormalize(dists, dataset.mu, dataset.st)
    samples = np.array(samples)
    samples_df = pd.DataFrame(data=samples, index=list(np.arange(0, len(samples))), columns=dataset.header_names[:-1])
    samples_df.to_csv(aug_samples_csv_path, index=False)


if __name__ == "__main__":
    main()
