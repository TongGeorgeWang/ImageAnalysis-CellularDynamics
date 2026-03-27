import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

import numpy as np
import scipy.signal
import scipy.optimize as opt
import matplotlib.animation as animation
from IPython.display import HTML
import importlib
import pandas as pd

import heapq
import tifffile as tf
import inspect
import itertools


from utils.spt import get_ROI_3, get_track_lims
from utils.helper import default_kwargs


def show_imag(image_arr, **kwargs):
    ''' 
    _Inputs:_
    * Can specify vmin and vmax by passing ``**get_val_range(SNR)`` as kwargs
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.imshow(image_arr, interpolation='none', **kwargs)
    fig.colorbar(img, ax=ax)
    return fig, ax

def show_imag_cbar(image_arr, **kwargs):
    ''' 
    _Inputs:_
    * Can specify vmin and vmax by passing ``**get_val_range(SNR)`` as kwargs
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.imshow(image_arr, interpolation='none', **kwargs)
    cbar = fig.colorbar(img, ax=ax)
    return fig, ax, cbar

def show_cropped_im(im_arr, col_xs, row_ys, **kwargs):
    ''' 
    * Handles incorrect indices. col_xs, row_ys can be out of bounds or not ints, will handle
    '''
    fig, ax = plt.subplots()

    n = im_arr.shape[0]
    for arr in [col_xs, row_ys]:
        for i in [0,1]:
            arr[i] = int(arr[i])
        arr[0] = max(arr[0], 0)
        arr[1] = min(n, arr[1])

    cropped_im = im_arr[row_ys[0]:row_ys[1], col_xs[0]:col_xs[1]]
    im = ax.imshow(cropped_im, extent=(col_xs[0], col_xs[1], row_ys[1], row_ys[0]), **kwargs)
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


def show_ROI(ax, col_xs, row_ys, **kwargs):
    if ('ec' not in kwargs) and ('edgecolor' not in kwargs):
        kwargs['ec'] = 'white'
    if 'fill' not in kwargs:
        kwargs['fill'] = False
    rect = patches.Rectangle((col_xs[0], row_ys[0]), col_xs[1]-col_xs[0], row_ys[1]-row_ys[0], **kwargs)
    ax.add_patch(rect)

def show_ROI_2(ax, col_xs, row_ys, box_kwargs = {}, marker_kwargs = {}, show_corners = True):
    '''
    * Note for kwargs: use abbreviated names, e.g. 'c' instead of 'color', 'ec' instead of 'edgecolor', 's' instead of 'size'
    '''
    
    box_default = {'ec':'white', 'fill':False, 'ls':'--'}
    marker_default = {'c':'white','marker':'*', 'lw':0.1, 's':60}
    box_default.update(box_kwargs)
    marker_default.update(marker_kwargs)

    rect = patches.Rectangle((col_xs[0], row_ys[0]), col_xs[1]-col_xs[0], row_ys[1]-row_ys[0], **box_default)
    ax.add_patch(rect)

    # markers on corners
    if show_corners:
        corners = list(itertools.product(col_xs, row_ys))
        xs, ys = zip(*corners)
        ax.scatter(xs, ys, **marker_default)


# wanna change the alpha_scatter idk ...
# also, can simply input fig and created the projection here but wtv
'''
def scatter_ROI(image_arr, col_xs, row_ys):

    cropped_im = image_arr[row_ys[0]:row_ys[1], col_xs[0]:col_xs[1]]
    n = image_arr.shape[0]
    x_im = np.arange(0, n)
    y_im = np.arange(0, n)
    x_crop = x_im[col_xs[0]:col_xs[1]]
    y_crop = y_im[row_ys[0]:row_ys[1]]

    fig, ax = alpha_scatter(cropped_im, x_crop, y_crop, rgb=(1, 0, 1)) # cuz it creates ax with projection 3d
    ax.set(xlabel='x', ylabel='y') #, title='fit')

    return fig, ax
'''

def profile_1d(im, axis, slice_loc, slice_range, **kwargs):
    '''
    TODO: fix kwargs \n
    * im: 2d array
    * axis: 0=vertical slice, 1=horizontal slice 
    * slice_loc: row index if horizontal slice, col index if vertical
    * slice_range: [col_x1, col_x2] if horizontal, [row_y1, row_y2] if vertical
    '''
    slice_loc = int(slice_loc)
    my_slice = (slice_loc, slice(None))[::(2*axis - 1)] # will reverse tuple if axis=0
    im_slice = im[my_slice][slice(*slice_range)]

    n_arr = np.arange(*slice_range)

    fig, ax = plt.subplots()
    ax.plot(n_arr, im_slice, marker='.', markersize=6, lw=1, c='black', **kwargs)
    # return im_slice
    return fig, ax

def profile_1d_v2(im, px_xy, lims, axis, **kwargs):
    '''
    TODO: make one that takes xy, lims as input and figures out axis on its own :)
    
    TODO: fix kwargs \n
    * im: 2d array
    * axis: 0=vertical slice, 1=horizontal slice 
    * slice_loc: row index if horizontal slice, col index if vertical
    * slice_range: [col_x1, col_x2] if horizontal, [row_y1, row_y2] if vertical
    '''
    # axis=1 -> px_xy[1], lims[0]
    # axis=0 -> px_xy[0], lims[1]
    slice_loc = int(px_xy[axis])
    slice_range = lims[not axis]
    my_slice = (slice_loc, slice(None))[::(2*axis - 1)] # will reverse tuple if axis=0
    im_slice = im[my_slice][slice(*slice_range)]

    n_arr = np.arange(*slice_range)

    fig, ax = plt.subplots()
    ax.plot(n_arr, im_slice, marker='.', markersize=6, lw=1, c='black', **kwargs)
    # return im_slice
    return fig, ax

def profile_1d_v2(im, px_xy, lims, axis, **kwargs):
    '''
    TODO: make one that takes xy, lims as input and figures out axis on its own :)
    
    TODO: fix kwargs \n
    * im: 2d array
    * axis: 0=vertical slice, 1=horizontal slice 
    * slice_loc: row index if horizontal slice, col index if vertical
    * slice_range: [col_x1, col_x2] if horizontal, [row_y1, row_y2] if vertical
    '''
    # axis=1 -> px_xy[1], lims[0]
    # axis=0 -> px_xy[0], lims[1]
    slice_loc = int(px_xy[axis])
    slice_range = lims[abs(1 - axis)]
    my_slice = (slice_loc, slice(None))[::(2*axis - 1)] # will reverse tuple if axis=0
    im_slice = im[my_slice][slice(*slice_range)]

    n_arr = np.arange(*slice_range)

    fig, ax = plt.subplots()
    ax.plot(n_arr, im_slice, **default_kwargs(kwargs, marker='.', markersize=6, lw=1, c='black'))
    # return im_slice
    return fig, ax



def profile_fit(f, popt, im, lims, axis=1, n_lin=100, fit_kwargs = {}, im_kwargs = {}):
    ''' 
    * can input int for lims: treated as ROI_width
    * axis: 0=vertical slice, 1=horizontal slice 
    '''

    if type(lims) == int:
        lims = get_ROI_3(*popt[0:2], ROI_width=lims, n=im.shape[0])

    loc = popt[axis]
    x_range = lims[abs(1-axis)]

    x = np.linspace(*x_range, n_lin)
    y = f(x, loc, *popt)

    fig, ax = profile_1d_v2(im, popt[0:2], lims, axis, **im_kwargs)
    ax.plot(x, y, **default_kwargs(fit_kwargs, c='red'))

    return fig, ax


def show_cropped_ROI(px_xy, ROI_width, im):
    n = im.shape[0]
    lims = get_ROI_3(*px_xy, ROI_width, n)
    fig_objs = show_cropped_im(im, *lims)
    return lims, fig_objs

def cycle_show_ROI(i, px_arr_i, ROI_width, im):
    return show_cropped_ROI(px_arr_i[i], ROI_width, im)




def implot_track_i(im, fp, stop, i_ptcl, offset = 5, line_color = 'white'):
    ''' 
    Plot i_ptcl's track on first frame, cropped.

    TODO: make the kwargs better ...
    '''

    track = fp[:stop[i_ptcl], i_ptcl, 0:2]
    t_lims = get_track_lims(track, offset)

    fig, ax, cbar = show_cropped_im(im, *t_lims, cmap='viridis')

    line = Line2D(track.T[0], track.T[1], c = line_color, lw=0.75) 
    ax.add_line(line)
    line.set_label(f'ptcl {i_ptcl}')

    ax.scatter(*track[0], c = 'black', marker='x', s=60, lw=2,  zorder=3) # show starting point

    return fig, ax, cbar




def output_frames(gen_frame_i, max_i, fpath):
    ''' 
    - assumes (for now) that gen_frame_i returns a tuple and the first element is a Figure type
    '''

    for i in range(max_i):
        print(i)
        outp = gen_frame_i(i)
        if (type(outp) == tuple) and (type(outp[0]) == mpl.figure.Figure):
            fig = outp[0]
        elif type(outp) == mpl.figure.Figure:
            fig = outp
        else:
            raise ValueError('wrong return type for gen_frame_i')
        
        fig.savefig(fpath + f'/frame_{i:04d}.png')
        plt.close(fig)



def plot_fit_param(fp, stop, i_ptcl, **params):
    ''' 
    - **params: e.g. w=2, A=3, B=4
    '''
    for name, i_param in params.items():
        fig, ax = plt.subplots()
        ax.plot(fp[:stop[i_ptcl], i_ptcl, i_param], c='black', lw=1)
        ax.set_title(name + ' vs t')
        ax.set(xlabel='Time steps', ylabel = name)

def plot_fit_param_1(fp, stop, i_ptcl, name, i_param, line_kwargs = {}):
    ''' 
    - **param: e.g. w=2. (only takes the first kwarg given)
    '''
    # name, i_param = list(param.items())[0]
    fig, ax = plt.subplots()
    ax.plot(fp[:stop[i_ptcl], i_ptcl, i_param], c='black', lw=1, **line_kwargs)
    ax.set_title(name + ' vs t')
    ax.set(xlabel='Time steps', ylabel = name)
    return fig, ax


def plot_fp_1(fp, stop,  **params):
    ''' 
    - **params: e.g. w=2, A=3, B=4
    '''
    for name, i_param in params.items():
        fig, ax = plt.subplots()
        ax.plot(fp[:stop, i_param], c='black', lw=1)
        ax.set_title(name + ' vs t')
        ax.set(xlabel='Time steps', ylabel = name)

def plot_fp_1_1(fp, stop,  name, i_param, line_kwargs = {}):
    ''' 
    - **param: e.g. w=2. (only takes the first kwarg given)
    '''
    # name, i_param = list(param.items())[0]
    fig, ax = plt.subplots()
    ax.plot(fp[:stop, i_param], c='black', lw=1, **line_kwargs)
    ax.set_title(name + ' vs t')
    ax.set(xlabel='Time steps', ylabel = name)
    return fig, ax



def GenFrame_OneTrack(ims, fp, stop_i, custom_ax = lambda x:x, im_kwarg = {}, **line_kwarg):

    def gen_frame_i(i):

        fig, ax = show_imag(ims[i], **im_kwarg)

        track = fp[:stop_i, 0:2]
        line = Line2D(track.T[0], track.T[1], **default_kwargs(line_kwarg, c = 'white', lw = 1))
        ax.add_line(line)

        custom_ax(ax)

        return fig, ax
    
    return gen_frame_i

def add_track(ax, fp, stop_i, **line_kwarg):
    track = fp[:stop_i, 0:2]
    line = Line2D(track.T[0], track.T[1], **default_kwargs(line_kwarg, c = 'white', lw = 1))
    ax.add_line(line)


def GenFrame_ScatterTrack(ims, fp, stop_i, custom_ax = lambda x:x, im_kwarg = {}, **scatter_kwarg):

    def gen_frame_i(i):

        fig, ax = show_imag(ims[i], **im_kwarg)

        track = fp[:stop_i, 0:2]
        ax.scatter(*track[i], **default_kwargs(scatter_kwarg, marker='x', c='black'))

        custom_ax(ax)

        return fig, ax
    
    return gen_frame_i




def GenFrame_CircTrack(ims, track, r=10, custom_ax = lambda x:x, im_kwarg={}, **circ_kwarg):
    '''
    '''
    def gen_frame_i(i):
        fig, ax = show_imag(ims[i], **im_kwarg)
        circ = patches.Circle(track[i], r, **default_kwargs(circ_kwarg, fill=False, ec='white'))
        ax.add_patch(circ)

        custom_ax(ax)

        return fig, ax
    
    return gen_frame_i