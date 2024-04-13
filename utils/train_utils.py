import torch
import re
import os
import cv2
import sys
import smtplib
import logging
import numpy as np

def copy_to_device(inputs, device, non_blocking=True):

    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: copy_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError("Unknown type: %s" % str(type(inputs)))
    return inputs

def copy_to_cuda(inputs, non_blocking=True):

    assert isinstance(inputs, dict), "inputs not dict"
    
    inputs = {k: v.cuda(non_blocking) for k, v in inputs.items()}

    return inputs

