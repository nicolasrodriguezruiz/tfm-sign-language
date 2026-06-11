"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

"""

Implementación de EarlyStopping

"""

import os
import time,random
import numpy as np
from collections import defaultdict, deque
import datetime

import torch


import pickle
import math

import torch.nn as nn
from torch import Tensor


#import tensorflow as tf

import matplotlib.pyplot as plt  # For graphics
import seaborn as sns


def is_main_process():
    return 'WORLD_SIZE' not in os.environ or os.environ['WORLD_SIZE']=='1' or os.environ['LOCAL_RANK']=='0'


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n


    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    if os.environ['LOCAL_RANK'] == '0':
                        if torch.cuda.is_available():
                            print(log_msg.format(
                                i, len(iterable), eta=eta_string,
                                meters=str(self),
                                time=str(iter_time), data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB))
                        else:
                            print(log_msg.format(
                                i, len(iterable), eta=eta_string,
                                meters=str(self),
                                time=str(iter_time), data=str(data_time)))
                else:
                    if torch.cuda.is_available():
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if os.environ['LOCAL_RANK'] == '0':
                print('{} Total time: {} ({:.4f} s / it)'.format(
                    header, total_time_str, total_time / len(iterable)))
        else:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))

def count_parameters_in_MB(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def load_dataset_file(filename):
    with open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 300
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]

def visualization(atten_maps):
    os.makedirs('./demo', exist_ok=True)
    for ii, att in enumerate(atten_maps):
        i = att.shape[0]
        idx = [max(1, int((i**0.5))), i//max(1, int((i**0.5))), 1]

        fig = plt.figure(figsize=(6*idx[1], 6*idx[0]))
        if att.squeeze().dim() == 2:
            ax = fig.add_subplot()
            att = torch.softmax(att, dim=-1)
            sns.heatmap(att.detach().cpu().numpy(), annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)

            fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
            plt.close()
            continue

        for cmp in att:
            ax = fig.add_subplot(*idx)
            sns.heatmap(cmp.detach().cpu().numpy(), cbar=idx[-1] % idx[-2] == 0, annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)
            idx[-1] += 1
        fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
        plt.close()

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, num_features=512, norm_type='sync_batch', num_groups=1):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            # raise ValueError("Please use sync_batch")
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == 'sync_batch':
            self.norm = nn.SyncBatchNorm(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=1,
        skip_connection=True):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.kernel_size = kernel_size
        if type(self.kernel_size)==int:
            conv_1 = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size, stride=1, padding='same')
            conv_2 = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size, stride=1, padding='same')
            self.pwff_layer = nn.Sequential(
                conv_1,
                nn.ReLU(),
                nn.Dropout(dropout),
                conv_2,
                nn.Dropout(dropout),
            )
        elif type(self.kernel_size)==list:
            pwff = []
            first_conv = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size[0], stride=1, padding='same')
            pwff += [first_conv, nn.ReLU(), nn.Dropout(dropout)]
            for ks in kernel_size[1:-1]:
                conv = nn.Conv1d(ff_size, ff_size, kernel_size=ks, stride=1, padding='same')
                pwff += [conv, nn.ReLU(), nn.Dropout(dropout)]
            last_conv = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size[-1], stride=1, padding='same')
            pwff += [last_conv, nn.Dropout(dropout)]

            self.pwff_layer = nn.Sequential(
                *pwff
            )
        else:
            raise ValueError
        self.skip_connection=skip_connection
        if not skip_connection:
            print('Turn off skip_connection in PositionwiseFeedForward')

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_t = x_norm.transpose(1,2)
        x_t = self.pwff_layer(x_t)
        if self.skip_connection:
            return x_t.transpose(1,2)+x
        else:
            return x_t.transpose(1,2)


class MLPHead(nn.Module):
    def __init__(self, embedding_size, projection_hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(nn.Linear(self.embedding_size, projection_hidden_size),
                                 nn.BatchNorm1d(projection_hidden_size),
                                 nn.ReLU(True),
                                 nn.Linear(projection_hidden_size, self.embedding_size))

    def forward(self, x):
        b, l, c = x.shape
        x = x.reshape(-1, c)  # x.view(-1,c)
        x = self.net(x)
        return x.reshape(b, l, c)  # x.view(b, l, c)

class EarlyStopping:
    """
    Para cuando la métrica de validación deja de mejorar.

    Opciones:
      mode:       'min' para WER/loss, 'max' para BLEU
      patience:   épocas sin mejora antes de parar
      min_delta:  mejora mínima que cuenta como progreso real
                  (evita parar por subidas de BLEU de 0.001)
    """
    def __init__(self, patience=10, mode='min', min_delta=0.0):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience} épocas sin mejora")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
