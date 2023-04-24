#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File  : mainflow.py
@Author: tao.jing
@Date  :
@Desc  :
"""
from __future__ import division
from __future__ import print_function

import ppynvml


def gpu_memory_log(logger=print, device=0):
    ppynvml.nvmlInit()
    handle = ppynvml.nvmlDeviceGetHandleByIndex(device)
    mem_info = ppynvml.nvmlDeviceGetMemoryInfo(handle)

    write_str = f'Used Memory: {float(mem_info.used / 1024 ** 2)} Mb'
    logger(write_str)
    write_str = f'Free Memory:{float(mem_info.free / 1024 ** 2)} Mb'
    logger(write_str)
    write_str = f'Total Memory: {float(mem_info.total / 1024 ** 2)} Mb'
    logger(write_str)

    ppynvml.nvmlShutdown()


def get_gpu_status(device=0):
    ppynvml.nvmlInit()
    handle = ppynvml.nvmlDeviceGetHandleByIndex(device)
    mem_info = ppynvml.nvmlDeviceGetMemoryInfo(handle)

    used_gpu_mb = float(mem_info.used / 1024 ** 2)
    free_gpu_mb = float(mem_info.free / 1024 ** 2)
    total_gpu_mb = float(mem_info.total / 1024 ** 2)

    ppynvml.nvmlShutdown()

    return {'used': used_gpu_mb, 'free': free_gpu_mb, 'total': total_gpu_mb}

if __name__ == '__main__':
    gpu_memory_log()
