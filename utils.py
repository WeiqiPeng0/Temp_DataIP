'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from shutil import get_terminal_size

import numpy as np
import matplotlib.pyplot as plt


def create_config(name):
    from configparser import ConfigParser

    #Get the configparser object
    config_object = ConfigParser()

    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["EXAMPLE"] = {
        "a": "something",
        "b": "place holder",
    }

    #Write the above sections to config.ini file
    with open(name, 'w') as conf:
        config_object.write(conf)

def read_config(name):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_object.read(name)
    return config_object



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width[0])
term_width, _ = get_terminal_size()

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def show_images(images, normalize=None, ipython=True,
                margin_height=2, margin_color='red',
                figsize=(18,16), save_npy=None):
    """ Shows pytorch tensors/variables as images """
    import matplotlib.pyplot as plt
    
    # first format the first arg to be hz-stacked numpy arrays
    if not isinstance(images, list):
        images = [images]
    images = [np.dstack(image.cpu().numpy()) for image in images]
    image_shape = images[0].shape
    assert all(image.shape == image_shape for image in images)
    assert all(image.ndim == 3 for image in images) # CxHxW

    # now build the list of final rows
    rows = []
    if margin_height >0:
        assert margin_color in ['red', 'black']
        margin_shape = list(image_shape)
        margin_shape[1] = margin_height
        margin = np.zeros(margin_shape)
        if margin_color == 'red':
            margin[0] = 1
    else:
        margin = None

    for image_row in images:
        rows.append(margin)
        rows.append(image_row)

    rows = [_ for _ in rows[1:] if _ is not None]
    plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    cat_rows = np.concatenate(rows, 1).transpose(1, 2, 0)
    imshow_kwargs = {}
    if cat_rows.shape[-1] == 1: # 1 channel: greyscale
        cat_rows = cat_rows.squeeze()
        imshow_kwargs['cmap'] = 'gray'

    plt.imshow(cat_rows, **imshow_kwargs)

    if save_npy is not None:
        scipy_img = scipy.misc.toimage(cat_rows)
        scipy_img.save(save_npy)

    plt.show()
    