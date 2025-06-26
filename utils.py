import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import subprocess
import torch.distributed as dist
import logging
from torch.distributions import Normal, Independent


def cross_entropy_loss(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone().float()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    labelf[labelf == 2] = 0
    cost = F.binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')

    return cost


def ELBO_loss(prediction, labelf, mean, std, beta):
    try:
        bce_loss = cross_entropy_loss(prediction, labelf, beta)
    except:
        import pdb;
        pdb.set_trace()
    var = torch.pow(std, 2)
    kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - torch.log(var))

    loss = bce_loss + 0.02 * kl_loss

    return loss


def uncertainty_loss(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone().float()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0

    uncertainty = 1 - 2 * torch.abs(prediction - 0.5)

    sigma = 0.2
    noedge = sigma * uncertainty + (1 - sigma) * prediction

    noedge_mask = torch.where(label == 0, torch.ones_like(label), torch.zeros_like(label))

    prediction = prediction * (1 - noedge_mask) + noedge * noedge_mask

    cost = F.binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')
    return cost


def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


logs = set()


def init_log(name, level=logging.INFO, filename=None):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fh = logging.FileHandler(filename)
    fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0

    # formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)4s] %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def step_lr_scheduler(optimizer, epoch, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch in lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    return optimizer


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # rank = dist.get_rank()
        # world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.cuda.set_device(rank % num_gpus)


    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return local_rank, rank, world_size


