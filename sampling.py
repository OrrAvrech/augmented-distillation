import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from models.transformer import BRT
from utils.data import Dataset, get_ds_type
import numpy as np
import pandas as pd


def sample_conditionals(data_loader, model, num_iter):
    conditionals = []
    for i in range(num_iter):
        for batch_idx, data in enumerate(data_loader):
            data_fetch = data.to(model.device)
            with torch.no_grad():
                batch_conditionals = model.batch_feed_forward(data_fetch)
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
    dataset_name = 'phoneme'
    data_dir = Path('./datasets/openml')
    batch_size = 32
    sampler = 'gibbs'
    gibbs_rounds = 1
    num_iter = 5
    model_path = Path('ck/2021_06_27_1136/phoneme_brt_2021_06_27_1136.pt')
    model_params_path = Path('ck/2021_06_27_1136/phoneme_brt_2021_06_27_1136.json')
    aug_samples_csv_path = data_dir / dataset_name / f"{dataset_name}_{sampler}_{num_iter}_1136.csv"

    dataset = get_ds_type(dataset_name, data_dir)
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
        dists = sample_conditionals(data_loader, model, num_iter)

    samples = denormalize(dists, dataset.mu, dataset.st)
    samples = np.array(samples)
    samples_df = pd.DataFrame(data=samples, index=np.arange(0, np.shape(samples)[0]), columns=dataset.header_names[:-1])
    samples_df.to_csv(aug_samples_csv_path, index=False)


if __name__ == "__main__":
    main()
