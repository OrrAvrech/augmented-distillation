import json
import copy
import random
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
from utils.data import Dataset, openml_checker, mixopml_checker, regsopml_checker
from models.transformer import BRT

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--anneal_learning_rate', default=False, action='store_true',
                    help='Whether to anneal the learning rate.')

parser.add_argument('--data_dir', default='./datasets/openml')
parser.add_argument('--datasetname', default='phoneme')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alg_name', type=str, default='brt')
parser.add_argument('--max_epochs', type=int, default=4, help='number of epochs to train (default: 1000)')
parser.add_argument('--disable_cuda', default=False, action='store_true')
parser.add_argument('--cuda_deterministic', default=False, action='store_true')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--set_num_threads', default=True, action='store_true', help='set_num_threads')

parser.add_argument('--log_id', default='dummy')
parser.add_argument('--check_point_dir', default='./ck')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--log_interval', type=int, default=50, help='log interval, one log per n updates')
parser.add_argument('--save_freq', type=int, default=250)
parser.add_argument('--eval_freq', default=5e3, type=float, help='How often (time steps) we evaluate')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers (default: 4)')

# ram/tam params
parser.add_argument('--hidden_size', type=int, default=100, help='hidden size (default: 100)')
parser.add_argument('--num_heads', type=int, default=5, help='num of multiattention head (default: 5)')
parser.add_argument('--num_layers', type=int, default=1, help='num of layers(default: 1)')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate(default: 0.1)')
parser.add_argument('--n_components', type=int, default=5, help='number of mixture component (default: 5)')
parser.add_argument('--max_gradient_norm', type=float, default=5, help='Max gradient norm')
parser.add_argument('--enable_tnsb', default=False, action='store_true', help='enable tensorboardX')
parser.add_argument('--save_test_pt', default=False, action='store_true', help='save test .pt')


def take_snapshot(args, ck_fname_part, bst_model, update, stats, save_test=False):
    """
        This function just save the current model and save some other info
    """
    fname_json = f"{ck_fname_part}.json"
    fname_pt = f"{ck_fname_part}.pt"

    print('Saving a checkpoint for iteration %d in %s' % (update, fname_json))
    checkpoint = {
                    'args': args.__dict__,
                 }
    for k, v in stats.items():
        checkpoint[k] = v
    with open(fname_json, 'w') as f:
        json.dump(checkpoint, f)

    # save the model
    if save_test:
        torch.save(bst_model.state_dict(), fname_pt)


def setup_log_checkpoints(log_dir, check_point_dir, datasetname, alg_name, log_id):

    check_point_dir.mkdir(parents=True, exist_ok=True)
    te_name = datasetname.split('_')[0]    # make sure to remove long name with _
    fname = f"{str.lower(te_name)}_{alg_name}_{log_id}"
    fname_log = log_dir / fname
    fname_eval = fname_log / 'eval.csv'

    return check_point_dir / fname, fname_log, fname_eval


def train(model, train_loader, optimizer, scheduler, global_step, params, device):
    """
        Train the model
    """
    model.train()
    train_loss = 0
    seen_so_far = 0
    mmd_loss_item = 0

    for batch_idx, data in enumerate(train_loader):

        ####
        # fetch data
        ####
        data = data.to(device)
        seen_so_far += data.shape[0]

        ####
        # loss cal
        ####
        d_data = data.shape[1]  # data is [B, D]
        optimizer.zero_grad()
        # nll: negative log likelihood
        logprobs = model.get_logprob(data) / d_data
        train_loss += logprobs.sum().item()
        nll = -logprobs.mean()

        ####
        # backward and optim
        ####
        nll.backward()

        # check if we clipping
        if params.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_gradient_norm)

        # run optim
        optimizer.step()

        ####
        # scheduler
        ####
        if scheduler is not None:
            scheduler.step()

        ####
        # log data
        ####
        if global_step % params.log_interval == 0 or global_step == 1:
            print('Train Log likelihood, step %d in nats: %.6f' % (global_step, train_loss / seen_so_far))

        global_step += 1

    print("Train epoch average loss: %f\n" % (train_loss/seen_so_far))
    return global_step, train_loss / seen_so_far


def evaluate(current_model, curr_loader, device, dset='Val'):
    """
        Test or evaluate current model
    """
    current_model.eval()
    val_loss = 0
    res_list = []

    for batch_idx, data in enumerate(curr_loader):

        ####
        # fetch data
        ####
        data = data.to(device)

        ####
        # get model performance
        ####
        with torch.no_grad():
            out_losses = current_model.get_logprob(data, use_all_dims=True)
            val_loss += out_losses.sum().item()
            res_list.extend(out_losses.cpu().numpy())

    print(dset + ',Log likelihood in nats: {:.6f}'.format(
            val_loss / len(curr_loader.dataset)))

    return val_loss / len(curr_loader.dataset), res_list


def main():
    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    ##############################
    # Generic setups
    ##############################
    CUDA_AVAL = torch.cuda.is_available()
    using_cuda = False

    if not args.disable_cuda and CUDA_AVAL:
        gpu_id = "cuda:0"  # + str(args.gpu_id)
        device = torch.device(gpu_id)
        using_cuda = True

    else:
        device = torch.device('cpu')
        print("**** No GPU detected or GPU usage is disabled, sorry! ****")

    if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.set_num_threads:
        torch.set_num_threads(1)

    ####
    # train and evalution checkpoints, log folders, ck file names
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # create folder for save checkpoints
    ckpt_dir = Path(args.check_point_dir)
    ck_fname_part, log_file_dir, fname_csv_eval = setup_log_checkpoints(log_dir, ckpt_dir,
                                                                        args.datasetname, args.alg_name, args.log_id)
    print(ck_fname_part, log_file_dir, fname_csv_eval)

    ##############################
    # Init dataset, model, alg, batch generator etc
    # Step 1: build dataset
    # Step 2: Build model
    # Step 3: Initiate Alg
    ##############################

    # Make results reproducible
    #  build_env already calls set seed,
    # Set seed the RNG for all devices (both CPU and CUDA)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ##############################
    # dataset builder/loader
    ##############################
    if openml_checker(args.datasetname)[0]:
        dataset = getattr(datasets, 'GENERIC')(args.data_dir, args.datasetname)

    elif mixopml_checker(args.datasetname)[0]:
        dataset = getattr(datasets, 'MIXDATA')(args.data_dir, args.datasetname)

    elif regsopml_checker(args.datasetname)[0]:
        dataset = getattr(datasets, 'REGSDATA')(args.data_dir, args.datasetname)

    else:
        raise ValueError(args.datasetname + " is not supported")

    args.train_size = dataset.train.N
    args.val_size = dataset.val.N
    dst_train = Dataset(data_obj=dataset.train)
    dst_valid = Dataset(data_obj=dataset.val)

    if using_cuda:
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

    else:
        kwargs = {}

    train_loader = DataLoader(dst_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)

    val_loader = DataLoader(dst_valid,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            **kwargs)

    ##############################
    # Build model/alg and optim
    ##############################
    if str.lower(args.alg_name) == 'brt':

        model = BRT(hidden_size=args.hidden_size,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    n_components=args.n_components,
                    device=device)

    else:
        raise ValueError("%s alg is not supported" % args.alg_name)

    print("Creating %s model.." % (str.upper(args.alg_name)))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # learning rate scheduler
    if args.anneal_learning_rate:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs * len(train_loader), 0)

    else:
        scheduler = None

    print(optimizer)
    print(scheduler)
    model.to(device)

    ##############################
    # Train and eval
    #############################
    # define some req vars
    epoch_counter = 0
    best_iter = 0
    best_val_loss = -1e10
    best_model = None
    tsb_writer = None

    stats_logs = {'train_losses': [],
                  'best_val_epoch': -1,
                  'val_losses': [],
                  'best_val_loss': -1,
                  'best_test_loss': -1,
                  'lr': [(0, args.lr)],
                  'best_val_mu_std': [],
                  }

    # just to keep params
    take_snapshot(args, ck_fname_part, model, 0, stats_logs)

    for epoch in range(args.max_epochs):
        print('\nEpoch: {}'.format(epoch))

        #######
        # train the model for one epoch
        #######
        epoch_counter, train_loss = train(model, train_loader, optimizer, scheduler, epoch_counter, args, device)

        #######
        # evaluate model on val dataset
        #######
        val_loss, val_llist = evaluate(model, val_loader, device)

        #######
        # logging
        #######
        stats_logs['val_losses'].append(val_loss)
        stats_logs['train_losses'].append(train_loss)

        if scheduler is not None:
            stats_logs['lr'].append((epoch, scheduler.get_lr()))

        #######
        # keep track of best model
        #######
        if val_loss > best_val_loss:
            best_iter = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            stats_logs['best_val_loss'] = best_val_loss
            stats_logs['best_val_epoch'] = best_iter
            stats_logs['best_val_mu_std'] = [np.mean(val_llist).item(), np.std(val_llist).item()]

            # take a snpshot and print
            if epoch > 1:  # and  args.save_freq % (epoch) == 0:
                save_this_time = True
                print("Saving best (current) val model at epoch %d " % epoch)

            else:
                save_this_time = False

            take_snapshot(args, ck_fname_part, best_model, epoch, stats_logs, save_test=save_this_time)
            print('Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
                  format(best_iter, best_val_loss))

    print('Done.')


if __name__ == "__main__":
    main()
