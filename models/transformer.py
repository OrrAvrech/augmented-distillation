from __future__ import print_function, division
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        """
            It encodes the position:
            PE(pos,2i)   = sin(pos/(10000)^(2i/d_model))
            PE(pos,2i+1) = cos(pos/(10000)^(2i/d_model))
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            X: [D, B, H] where D is seq length,
            B is batch size, H is embedding size
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BRT(nn.Module):

    def __init__(self, input_size=1,
                 output_size=1,
                 hidden_size=25,
                 num_heads=5,
                 num_layers=1,
                 dropout=0.1,
                 n_components=1,
                 s_act='identity',
                 max_len_pos=1000,
                 device='cpu'):
        """
            This model use Transformer Autoregressive Model (TAM)
        """
        super(BRT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # MultiHeadAttention params
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_ff = hidden_size
        self.d_model = hidden_size
        self.num_layers = num_layers

        # Gaussian Mixture components
        self.n_components = n_components

        # embedding layer
        # activation function for embedding layer
        activations = {'identity': nn.Identity}
        act_func = activations[s_act]

        # embedding layer
        self.fc_emb = nn.Sequential(
            nn.Linear(self.input_size, self.d_model),
            act_func()
        )
        # pos embd layer
        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              dropout=dropout,
                                              max_len=max_len_pos
                                              )

        # input size would be 1 as we roll out X [B,D] ==> [B,D,1], treat D dim as seq length
        transformer_block = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=num_heads,
                                                       dim_feedforward=self.d_model,
                                                       dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_block, num_layers=num_layers)
        # output layer
        self.fc_mean = nn.Linear(self.d_model, self.n_components * self.output_size)
        self.log_std = nn.Linear(self.d_model, self.n_components * self.output_size)

        if self.n_components > 1:
            self.mixing_frac = nn.Linear(self.d_model, self.n_components)

        # gpu or cpu device
        self.device = device

    def forward(self, xin, mid=np.inf, h=None):
        """
             input :
                    X: [B, D], B batch size, and D is Dim
                    h: [num_layers, B, hidden_dim]
             return:
                    mu: [B, D]
                    log_std: [B, D]
        """
        # step 0: reshape [B, D] ==> [B, D, 1]
        x = xin.view(xin.shape[0], xin.shape[1], 1)
        seq_len = xin.shape[1]

        # step 1: reshape [B, D, 1] ==> [B, D, d_model/hidden_size]# D is consider as seq length
        x = self.fc_emb(x)
        x[:, mid:mid + 1, :] = 0

        # step 2: run through transformer block
        # [B, D, d_model/hidden_size] ==> [B, D, d_model/hidden_size]
        # transfomer block accepts [D, B, d_model/hidden_size]
        x = x.transpose(0, 1)  # [B, D/seq_len, d_model/hidden_size] ==> [D/seq_len, B, d_model/hidden_size]
        # build the masking ==> [seq/D, seq/D]
        m_temp = torch.ones(seq_len, seq_len)
        m_temp[mid, mid] = 0
        mask = (m_temp == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(self.device)

        # step 3: apply position encoding
        pos_out = self.pos_encoder(x * math.sqrt(self.d_model))

        # feed data to the transformer
        trans_out = self.transformer_encoder(pos_out, mask=mask)  # [D/seq_len, B, d_model/hidden_size]
        trans_out = trans_out.transpose(0, 1)   # [D/seq_len, B, d_model/hidden_size] ==>
        # [B, D/seq_len, d_model/hidden_size]

        # step 4: [B, Seq, hidden_dim] ==> [B, D, 1] ==> [B, D]
        # Note if self.n_components > 1, then [B, D, C], squeeze(-1) doesn't affect
        mu = self.fc_mean(trans_out).squeeze(-1)
        a = self.log_std(trans_out).squeeze(-1)

        # return mixing fractions
        if self.n_components > 1:
            pi = self.mixing_frac(trans_out).squeeze(-1)

        else:
            pi = None

        return mu, a, h, pi

    def compute_conditionals(self, mu, a, pi, i):
        if self.n_components == 1:

            # generate sample xi =  mu + eps * a.exp()
            # [B, i+1] ==> [B, 1]
            mu, a = mu[:, i:i + 1], a[:, i:i + 1]
            cat = torch.distributions.Normal(mu, a.exp())
            batch_samples = cat.sample().squeeze(-1)  # [B, 1] ==> [B]

        else:
            # [B, i+1 , C] ==> [B, 1 , C]
            mu, a, pi = mu[:, i:i + 1, :], a[:, i:i + 1, :], pi[:, i:i + 1, :]

            # normalize pi ([B, 1 , C] - [B, 1, 1] ==> [B, i+1 , C])
            # .squeeze(-2) ==> [B, 1 , C] ==> [B, C]
            logpi = (pi - pi.logsumexp(-1, keepdim=True)).squeeze(-2)

            # idx ==> [B, C] ==> [B] ==> unsqueeze ==> [B, 1]
            idxz = torch.distributions.Categorical(logits=logpi).sample().unsqueeze(-1)

            # mu [B, 1, C] ==> squeeze(-2) ==> [B, C]
            mu_z = torch.gather(mu.squeeze(-2), 1, idxz)  # [B, 1]
            au_z = torch.gather(a.squeeze(-2), 1, idxz)  # [B, 1]

            # now sample
            cat = torch.distributions.Normal(mu_z, au_z.exp())
            batch_samples = cat.sample().squeeze(-1)  # [B, 1] ==> [B]

        return batch_samples

    def get_logprob(self, xin, use_all_dims=False):
        """
          This function returns -logprobs
          xin: [B, D], B batch size, and D is Dim
        """
        logprobs = []

        if use_all_dims:
            loop_indx = range(xin.shape[1])

        else:
            loop_indx = [np.random.randint(xin.shape[1])]

        for mid in loop_indx:
            xin_c = xin.clone()
            xin_c[:, mid] = 0

            # step 1: call forward function
            mu, a, _, pi = self(xin_c, mid=mid)  # both are [B, D]

            if self.n_components == 1:
                # step 1: get p(xi | x1:i−1) = N(xi | µi, (expαi))
                mu = mu[:, mid:mid + 1]  # [B, D] ==> [B, 1]
                a = a[:, mid:mid + 1]  # [B, D] ==> [B, 1]
                cat = torch.distributions.Normal(mu, a.exp())

                # step 2: calculate log_prob
                logprobs00 = cat.log_prob(xin[:, mid:mid + 1])  # sum([B, D]) ==> [B, 1]
                logprobs.append(logprobs00.squeeze(-1))

            else:
                B = xin.shape[0]  # batch size
                D = xin.shape[1]  # dim
                C = self.n_components  # number of component

                # [B, D] ==> repeat ==> [B, C * D] ==> view ==> [B, C, D]
                xin_rp = xin.repeat(1, C).view(B, C, D)

                # log prob
                mu = (mu.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
                a = (a.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
                pi = (pi.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
                cat = torch.distributions.Normal(mu, a.exp())
                logprobs0 = (cat.log_prob(xin_rp))[:, :, mid:mid + 1]  # logprobs [B, C, D]

                # normalize pi ([B, C , D] - [B, 1 , D] ==> [B, C , D])
                logpi = pi - pi.logsumexp(1, keepdim=True)

                # log(\sum (pi  * N(mu, sigma) )) [B, C , D] ==> [B, D]
                logprobs00 = torch.logsumexp(logpi + logprobs0, dim=1)
                logprobs.append(logprobs00.squeeze(-1))

                # .sum(-1, keepdim=True) [B, D] ==> [B, 1]
        logprobs = torch.stack(logprobs, dim=-1)
        logprobs = logprobs.sum(-1, keepdim=True)

        return logprobs

    def gibbs_sampler(self, rounds, real_data, dim_orders, permute_dims=False, random_init=False):
        """
            Generate data using gibbs sampler.
            rounds: number of gibbs sampler round
            real_data: B * D
            dim_orders: is a list of how to sample from different, default [0,..D]
            permute_dims: if is ture, it permutes dim_orders
        """
        # step 0: find deivce type e.g. cpu or gpu
        device = next(self.parameters()).device
        out_per_round = []

        # step 1:
        # noise: B * D
        if random_init:
            noise = torch.rand(real_data.shape[0], real_data.shape[1]).type(real_data.dtype).to(device)

        else:
            noise = real_data.clone()

        with torch.no_grad():

            for _ in range(rounds):  # gibbs sampler rounds

                # step 2: generate D samples where D is data dim
                for i in dim_orders:

                    # setp 3: call forward function
                    # since every single time **ALL** samples are required,
                    # noise should be B * D
                    noise_temp = noise.clone()
                    noise_temp[:, i] = 0
                    mu, a, _, pi = self(noise_temp, mid=i)  # mu, a, pi ==> [B, D, C]
                    batch_samples = self.compute_conditionals(mu, a, pi, i)
                    noise[:, i] = batch_samples

                # keep results per round
                out_per_round.append(noise.clone())

                if permute_dims:
                    random.shuffle(dim_orders)

        # out is a list with len round, and each item is B * D
        # torch.stack(out, dim = 0)  ==> [R, B, D]
        out_per_round = torch.stack(out_per_round, dim=0)

        return noise, out_per_round

    def batch_feed_forward(self, input_data, mid):
        samples = input_data.clone()
        mu, a, _, pi = self(input_data, mid=mid)
        batch_samples = self.compute_conditionals(mu, a, pi, mid)
        samples[:, mid] = batch_samples
        return samples

    def fit(self, train_loader, val_loader):
        """
            Train the model
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # learning rate scheduler
        if self.anneal_learning_rate:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(train_loader), 0)
        else:
            scheduler = None

        # Tensorboard writer
        tb_writer = SummaryWriter(log_dir=self.log_dir)

        train_loss = 0
        seen_so_far = 0
        global_step = 0
        best_val_loss = -1e10
        stats_logs = {'train_losses': [],
                      'best_val_epoch': -1,
                      'val_losses': [],
                      'best_val_loss': -1,
                      'best_test_loss': -1,
                      'lr': [(0, self.lr)],
                      'best_val_mu_std': [],
                      }

        for epoch in range(self.epochs):
            print(f"\nEpoch: {epoch}")

            self.train()
            for batch_idx, data in enumerate(train_loader):

                ####
                # fetch data
                ####
                data = data.to(self.device)
                seen_so_far += data.shape[0]

                ####
                # loss cal
                ####
                d_data = data.shape[1]  # data is [B, D]
                optimizer.zero_grad()
                # nll: negative log likelihood
                logprobs = self.get_logprob(data) / d_data
                train_loss += logprobs.sum().item()
                nll = -logprobs.mean()

                ####
                # backward and optim
                ####
                nll.backward()

                # check if we clipping
                if self.max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)

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
                if global_step % self.log_interval == 0 or global_step == 1:
                    print(f"Train Log likelihood, step {global_step} in nats: {train_loss / seen_so_far:.6f}")

                global_step += 1
            print(f"Train epoch average loss: {train_loss / seen_so_far}\n")

            self.eval()
            val_loss = 0
            res_list = []
            for val_batch_idx, val_data in enumerate(val_loader):
                ####
                # fetch data
                ####
                val_data = val_data.to(self.device)

                ####
                # get model performance
                ####
                with torch.no_grad():
                    out_losses = self.get_logprob(val_data, use_all_dims=True)
                    val_loss += out_losses.sum().item()
                    res_list.extend(out_losses.cpu().numpy())
            print(f"Validation Log likelihood in nats: {val_loss / len(val_loader.dataset):.6f}")

            #######
            # logging
            #######
            stats_logs['val_losses'].append(val_loss)
            stats_logs['train_losses'].append(train_loss)

            if scheduler is not None:
                stats_logs['lr'].append((epoch, scheduler.get_lr()))
                tb_writer.add_scalar('Learning-rate', scheduler.get_last_lr()[0], epoch)
            else:
                tb_writer.add_scalar('Learning-rate', optimizer.param_groups[0]['lr'], epoch)

            # write to Tensorboard
            tb_writer.add_scalar('Loss/train', train_loss, epoch)
            tb_writer.add_scalar('Loss/validation', val_loss, epoch)

            if val_loss > best_val_loss:
                best_iter = epoch
                best_val_loss = val_loss
                stats_logs['best_val_loss'] = best_val_loss
                stats_logs['best_val_epoch'] = best_iter
                stats_logs['best_val_mu_std'] = [np.mean(res_list).item(), np.std(res_list).item()]
                print("Saving best (current) val model at epoch %d " % epoch)
                take_snapshot(args, ck_fname_part, best_model, epoch, stats_logs, save_test=save_this_time)
                print('Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
                      format(best_iter, best_val_loss))

# class LitConditionalTransformer(pl.LightningModule):
#
#     def __init__(self, input_size=1,
#                  output_size=1,
#                  hidden_size=25,
#                  num_heads=5,
#                  num_layers=1,
#                  dropout=0.1,
#                  n_components=1,
#                  s_act='identity',
#                  max_len_pos=1000):
#         super(LitConditionalTransformer, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#
#         # MultiHeadAttention params
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.d_ff = hidden_size
#         self.d_model = hidden_size
#         self.num_layers = num_layers
#
#         # Gaussian Mixture components
#         self.n_components = n_components
#
#         # embedding layer
#         # activation function for embedding layer
#         activations = {'identity': nn.Identity}
#         act_func = activations[s_act]
#
#         # embedding layer
#         self.fc_emb = nn.Sequential(
#             nn.Linear(self.input_size, self.d_model),
#             act_func()
#         )
#         # pos embd layer
#         self.pos_encoder = PositionalEncoding(d_model=self.d_model,
#                                               dropout=dropout,
#                                               max_len=max_len_pos
#                                               )
#
#         # input size would be 1 as we roll out X [B,D] ==> [B,D,1], treat D dim as seq length
#         transformer_block = nn.TransformerEncoderLayer(d_model=self.d_model,
#                                                        nhead=num_heads,
#                                                        dim_feedforward=self.d_model,
#                                                        dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(transformer_block, num_layers=num_layers)
#         # output layer
#         self.fc_mean = nn.Linear(self.d_model, self.n_components * self.output_size)
#         self.log_std = nn.Linear(self.d_model, self.n_components * self.output_size)
#
#         if self.n_components > 1:
#             self.mixing_frac = nn.Linear(self.d_model, self.n_components)
#
#     def forward(self, xin, mid=np.inf, h=None):
#         """
#              input :
#                     X: [B, D], B batch size, and D is Dim
#                     h: [num_layers, B, hidden_dim]
#              return:
#                     mu: [B, D]
#                     log_std: [B, D]
#         """
#         # step 0: reshape [B, D] ==> [B, D, 1]
#         x = xin.view(xin.shape[0], xin.shape[1], 1)
#         seq_len = xin.shape[1]
#
#         # step 1: reshape [B, D, 1] ==> [B, D, d_model/hidden_size]# D is consider as seq length
#         x = self.fc_emb(x)
#         x[:, mid:mid + 1, :] = 0
#
#         # step 2: run through transformer block
#         # [B, D, d_model/hidden_size] ==> [B, D, d_model/hidden_size]
#         # transfomer block accepts [D, B, d_model/hidden_size]
#         x = x.transpose(0, 1)  # [B, D/seq_len, d_model/hidden_size] ==> [D/seq_len, B, d_model/hidden_size]
#         # build the masking ==> [seq/D, seq/D]
#         m_temp = torch.ones(seq_len, seq_len)
#         m_temp[mid, mid] = 0
#         mask = (m_temp == 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         mask = mask.to(self.device)
#
#         # step 3: apply position encoding
#         pos_out = self.pos_encoder(x * math.sqrt(self.d_model))
#
#         # feed data to the transformer
#         trans_out = self.transformer_encoder(pos_out, mask=mask)  # [D/seq_len, B, d_model/hidden_size]
#         trans_out = trans_out.transpose(0, 1)  # [D/seq_len, B, d_model/hidden_size] ==>
#         # [B, D/seq_len, d_model/hidden_size]
#
#         # step 4: [B, Seq, hidden_dim] ==> [B, D, 1] ==> [B, D]
#         # Note if self.n_components > 1, then [B, D, C], squeeze(-1) doesn't affect
#         mu = self.fc_mean(trans_out).squeeze(-1)
#         a = self.log_std(trans_out).squeeze(-1)
#
#         # return mixing fractions
#         if self.n_components > 1:
#             pi = self.mixing_frac(trans_out).squeeze(-1)
#
#         else:
#             pi = None
#
#         return mu, a, h, pi
#
#     def get_logprob(self, xin, use_all_dims=False):
#         """
#           This function returns -logprobs
#           xin: [B, D], B batch size, and D is Dim
#         """
#         logprobs = []
#
#         if use_all_dims:
#             loop_indx = range(xin.shape[1])
#
#         else:
#             loop_indx = [np.random.randint(xin.shape[1])]
#
#         for mid in loop_indx:
#             xin_c = xin.clone()
#             xin_c[:, mid] = 0
#
#             # step 1: call forward function
#             mu, a, _, pi = self(xin_c, mid=mid)  # both are [B, D]
#
#             if self.n_components == 1:
#                 # step 1: get p(xi | x1:i−1) = N(xi | µi, (expαi))
#                 mu = mu[:, mid:mid + 1]  # [B, D] ==> [B, 1]
#                 a = a[:, mid:mid + 1]  # [B, D] ==> [B, 1]
#                 cat = torch.distributions.Normal(mu, a.exp())
#
#                 # step 2: calculate log_prob
#                 logprobs00 = cat.log_prob(xin[:, mid:mid + 1])  # sum([B, D]) ==> [B, 1]
#                 logprobs.append(logprobs00.squeeze(-1))
#
#             else:
#                 B = xin.shape[0]  # batch size
#                 D = xin.shape[1]  # dim
#                 C = self.n_components  # number of component
#
#                 # [B, D] ==> repeat ==> [B, C * D] ==> view ==> [B, C, D]
#                 xin_rp = xin.repeat(1, C).view(B, C, D)
#
#                 # log prob
#                 mu = (mu.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
#                 a = (a.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
#                 pi = (pi.transpose(-2, -1))[:, :, mid:mid + 1]  # [B, D, C] ==> [B, C, D]
#                 cat = torch.distributions.Normal(mu, a.exp())
#                 logprobs0 = (cat.log_prob(xin_rp))[:, :, mid:mid + 1]  # logprobs [B, C, D]
#
#                 # normalize pi ([B, C , D] - [B, 1 , D] ==> [B, C , D])
#                 logpi = pi - pi.logsumexp(1, keepdim=True)
#
#                 # log(\sum (pi  * N(mu, sigma) )) [B, C , D] ==> [B, D]
#                 logprobs00 = torch.logsumexp(logpi + logprobs0, dim=1)
#                 logprobs.append(logprobs00.squeeze(-1))
#
#                 # .sum(-1, keepdim=True) [B, D] ==> [B, 1]
#         logprobs = torch.stack(logprobs, dim=-1)
#         logprobs = logprobs.sum(-1, keepdim=True)
#
#         return logprobs
#
#     def training_step(self, batch, batch_idx):
#         d_data = batch.shape[1]  # data is [B, D]
#         # nll: negative log likelihood
#         logprobs = self.get_logprob(batch) / d_data
#         nll = -logprobs.mean()
#         self.log('train_loss', nll)
#         return nll
#
#     def configure_optimizers(self):



