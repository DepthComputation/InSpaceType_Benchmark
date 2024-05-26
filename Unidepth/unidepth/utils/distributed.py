"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import platform
import warnings
import subprocess

import cv2

import torch
import torch.utils.data.distributed
from torch import multiprocessing as mp
from torch import distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def barrier():
    if not is_dist_avail_and_initialized():
        return
    dist.barrier()


def is_main_process():
    return get_rank() == 0


def is_rank_zero(args):
    return args.rank == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != "Windows":
        mp_start_method = cfg.get("mp_start_method", "fork")
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f"Multi-processing start method `{mp_start_method}` is "
                f"different from the previous setting `{current_method}`."
                f"It will be force set to `{mp_start_method}`. You can change "
                f"this behavior by changing `mp_start_method` in your config."
            )
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get("opencv_num_threads", 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = cfg.get("workers_per_gpu", 4)

    if "OMP_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f"Setting OMP_NUM_THREADS environment variable for each process "
            f"to be {omp_num_threads} in default, to avoid your system being "
            f"overloaded, please further tune the variable for optimal "
            f"performance in your application as needed."
        )
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # setup MKL threads
    if "MKL_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = os.environ.get("OMP_NUM_THREADS", 1)
        warnings.warn(
            f"Setting MKL_NUM_THREADS environment variable for each process "
            f"to be {mkl_num_threads} in default, to avoid your system being "
            f"overloaded, please further tune the variable for optimal "
            f"performance in your application as needed."
        )
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)


def setup_slurm(backend: str, port: str) -> None:
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]

    num_gpus = torch.cuda.device_count()

    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)
    print(
        proc_id,
        ntasks,
        num_gpus,
        proc_id % num_gpus,
        node_list,
        addr,
        os.environ["MASTER_PORT"],
        os.system("nvidia-smi -L"),
    )
    dist.init_process_group(backend, rank=proc_id, world_size=ntasks)


def sync_tensor_across_gpus(t, dim=0, cat=True):
    if t is None or not (dist.is_available() and dist.is_initialized()):
        return t
    t = torch.atleast_1d(t)
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)

    local_size = torch.tensor(t.size(dim), device=t.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(group_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)
    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=t.device, dtype=t.dtype)
        t = torch.cat((t, padding))

    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]
    dist.all_gather(gather_t_tensor, t)
    all_ts = []
    for t, size in zip(gather_t_tensor, all_sizes):
        all_ts.append(t[:size])
    if cat:
        return torch.cat(all_ts, dim=0)
    return all_ts


import pickle


def sync_string_across_gpus(keys, device, dim=0):
    keys_serialized = pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL)
    keys_serialized_tensor = torch.frombuffer(keys_serialized, dtype=torch.uint8).to(
        device
    )
    keys_serialized_tensor = sync_tensor_across_gpus(
        keys_serialized_tensor, dim=0, cat=False
    )
    keys = [
        key
        for keys in keys_serialized_tensor
        for key in pickle.loads(bytes(keys.cpu().tolist()))
    ]
    return keys
