# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：GPU_manager_pytorch.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024-03-23 10:10:06
"""
from pynvml import *
import torch
from torch.autograd import Variable
import math

# the default GPU
GPU_DEVICE_ID = torch.cuda.device_count() - 1
import psutil
import time
import random


def get_free_memory_percentage():
    virtual_memory = psutil.virtual_memory()
    # free_memory_percentage = virtual_memory.free / virtual_memory.total * 100
    free_memory_percentage = virtual_memory.available / virtual_memory.total * 100

    return free_memory_percentage


def check_memory(free_thre=5, max_attempts=100000000):
    attempts = 0
    while attempts < max_attempts:
        free_memory_percentage = get_free_memory_percentage()
        print(f'Free Memory Percentage: {free_memory_percentage:.2f}%')

        if free_memory_percentage > free_thre:
            return free_memory_percentage
        wait_seconds = random.randint(10, 100)
        print('Waiting for free memory to exceed {}%, wait {}s...'.format(free_thre, wait_seconds))

        time.sleep(wait_seconds)
        attempts += 1
    raise MyCustomError(f"{max_attempts} times try check free memory, but fail.")


class MyCustomError(Exception):
    def __init__(self, message):
        self.message = message


def auto_select_gpu_and_cpu(free_thre=5, max_attempts=100000000):
    """
    Automatically select the gpu device with the max available memory space
    :return: the device id
    """
    if not torch.cuda.is_available():
        raise MyCustomError("No aviable cuda")
    attempts = 0
    while attempts < max_attempts:
        global GPU_DEVICE_ID
        n_device = torch.cuda.device_count()
        max_memory = -math.inf
        total_memory = -math.inf

        nvmlInit()
        for i in range(min(n_device, 8)):
            device_id = n_device - 1 - i
            h = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(h)
            free_memory_space = int(info.free)
            total_memory_space = int(info.total)
            print(f'[INFO] GPU device { device_id} - total: {round(total_memory_space/ (1024 ** 3),3)}GB; - memory free: {round(free_memory_space/ (1024 ** 3),3)}GB.')
            if free_memory_space > max_memory:
                max_memory = free_memory_space
                total_memory = total_memory_space
                GPU_DEVICE_ID = device_id
        if total_memory * (free_thre / 100) < max_memory:
            print(f'\n[INFO] more than {free_thre}% free memory, Auto select GPU device {GPU_DEVICE_ID},- memory free: {round(max_memory/ (1024 ** 3),3)}GB')
            return "cuda:" + str(GPU_DEVICE_ID)
        wait_seconds = random.randint(10, 100)
        print('\n[INFO] No more than {}% free memory on every GPU, so wait {}s '.format(free_thre, wait_seconds))
        time.sleep(wait_seconds)
        attempts += 1
    raise MyCustomError(f"{max_attempts} times try find GPU, but fail.")
def check_gpu_memory(device_id,free_thre=5, max_attempts=100000000):
    """
    check GPU memory
    :return: the device id
    """
    if not torch.cuda.is_available():
        raise MyCustomError("No aviable cuda")
    attempts = 0
    while attempts < max_attempts:
        h = nvmlDeviceGetHandleByIndex(device_id)
        info = nvmlDeviceGetMemoryInfo(h)
        free_memory_space = int(info.free)
        total_memory_space = int(info.total)
        print('[INFO] GPU device ', device_id, '- memory free: ', free_memory_space)
        if free_memory_space > total_memory_space * (free_thre / 100):
            print('\n[INFO] more than {}% free memory on device {},- memory free: {}'.format(free_thre, device_id,free_memory_space))
            return
        wait_seconds = random.randint(10, 100)
        print('\n[INFO] No more than {}% free memory on device {}, so wait {}s '.format(free_thre,device_id, wait_seconds))
        time.sleep(wait_seconds)
        attempts += 1
    raise MyCustomError(f"{max_attempts} times try find GPU, but fail.")

def auto_select_gpu(min_mem_pct=0.1):
    """自动选择 GPU"""
    while True:
        nvmlInit()
        mem_info = [(i, 100 * nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free / nvmlDeviceGetMemoryInfo(
            nvmlDeviceGetHandleByIndex(i)).total) for i in range(torch.cuda.device_count())]

        mem_info = sorted(mem_info, key=lambda x: x[1], reverse=True)
        print('[INFO] GPU device - memory free%: ', mem_info)
        if mem_info[0][1] >= 10:
            return "cuda:" + str(mem_info[0][0])
        else:
            print(f"No GPU with enough memory available. Waiting for 10 minutes...")
            time.sleep(60 * 10)


def get_device_id():
    return GPU_DEVICE_ID


def get_device():
    return torch.device("cuda:" + str(GPU_DEVICE_ID) if torch.cuda.is_available() else "cpu")


def log_sum_exp(value, dim=None, keep_dim=False):
    """
    Numerically stable implementation of the operation
    value.exp().sum(dim, keep_dim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keep_dim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keep_dim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, (int, float, complex)):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def running_average_tensor_list(first_list, second_list, rate):
    """
    Return the result of
    first_list * (1 - rate) + second_list * rate
    Parameter
    ---------
    first_list (list) : A list of pytorch Tensors
    second_list (list) : A list of pytorch Tensors, should have the same
        format as first_list.
    rate (float): a learning rate, in [0, 1]
    Returns
    -------
    results (list): A list of Tensors with computed results.
    """
    results = []
    assert len(first_list) == len(second_list)
    for first_t, second_t in zip(first_list, second_list):
        assert first_t.shape == second_t.shape
        result_tensor = first_t * (1 - rate) + second_t * rate
        results.append(result_tensor)
    return results


def constant(value):
    """
    Return a torch Variable for computation. This is a function to help
    write short code.
    pytorch require multiplication take either two variables
    or two tensors. And it is recommended to wrap a constant
    in a variable which is kind of silly to me.
    https://discuss.pytorch.org/t/adding-a-scalar/218
    Parameters
    ----------
    value (float): The value to be wrapped in Variable
    Returns
    -------
    constant (Variable): The Variable wrapped the value for computation.
    """
    # noinspection PyArgumentList
    return Variable(torch.Tensor([value])).type(torch.float)


def freeze_parameters(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = False


def unfreeze_parameters(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = True


def get_parameters(module_list):
    parameters = []
    for module in module_list:
        parameters.extend(list(module.parameters()))
    return parameters


def binarize_tensor(tensor, threshold):
    tensor[tensor >= threshold] = 1.0
    tensor[tensor < threshold] = 0.0
    return tensor


def tensors_to_numpy(tensors):
    np_tensors = []
    for tensor in tensors:
        np_tensors.append(tensor.detach().cpu().numpy())
    return np_tensors


def identity(x, dim=0):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x
def monitor_resources(GPU_index=0):
    import psutil
    import pynvml
    # 获取当前进程
    pid = os.getpid()
    p = psutil.Process(pid)

    # CPU利用率（相对于单个核心）
    cpu_usage = p.cpu_percent(interval=1) / psutil.cpu_count()

    # 内存占用
    memory_info = p.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # 将字节转换成MB

    # 初始化GPU监控
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_index)  # 假设只监控第一个GPU
    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_mem_usage = (gpu_mem_info.used / gpu_mem_info.total) * 100  # GPU内存使用率
    pynvml.nvmlShutdown()

    return cpu_usage, memory_usage, gpu_usage, gpu_mem_usage