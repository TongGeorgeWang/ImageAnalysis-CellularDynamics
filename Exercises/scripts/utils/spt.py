import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.animation as animation
from IPython.display import HTML, display, Math
import importlib
import pandas as pd

import heapq
import tifffile as tf
import inspect
import itertools

# 07-15
# temporary ...
# iguess there is the issue of all the things in helper3 now being available directly in spt.(...)
# so ideally... only import what's necessary?
# there u go...
# can modify visibility in init idk later

from utils.helper import plot_fit_3d_v2, plot_fit_3d_v3, px2pos, ricker2d, slice2d # , ExpInfo
from utils.helper import default_kwargs
from utils.helper import ExpInfo # seems to work fine!
# 2025-07-29: added plot_MSD_2 which requires ExpInfo: unsure if this will work but...
# import helper3


def select_ROI(col_x, row_y, ROI_radius, n, im_to_show = np.zeros(0)):
    '''
    Inputs:
    * col_x, row_y: the (x,y) pixel-position of the center of the ROI (dont have to be ints) \n
    *Returns*: \n
    * col_xs: the left and right column/x indices of the ROI
    * row_ys: the up and down row/y indices of the ROI
    * corners: the (row, col) tuples to index the image array (only relevant for displaying the corners... might omit) \n
    *Comments*: \n
    - workaround... make it so if dont put im_to_show, it wont show ... so use np.zeros(0) \n
    - dont wanna input noise info... (i could re-make image from pos_arr but..) so just do vmax=max val \n
    '''
    d = int(ROI_radius)
    
    # the x,y indices delimiting the ROI
    # 06-16: added int()
    col_xs = [int(max(0, col_x-d)), int(min(n-1, col_x+d))]
    row_ys = [int(max(0, row_y-d)), int(min(n-1, row_y+d))]

    corners = [
        ## ahh! indexing an ARRAY is (row, col)
        (row_ys[0], col_xs[0]),
        (row_ys[0], col_xs[1]),
        (row_ys[1], col_xs[0]),
        (row_ys[1], col_xs[1])
    ]

    if im_to_show.size > 0:
        val = im_to_show.max()*1.5
        im_copy = np.copy(im_to_show)
        for coord in corners:
            im_copy[coord] = val
        fig, ax = plt.subplots()
        im = ax.imshow(im_copy, cmap='viridis', vmin=0, vmax=im_to_show.max())
        plt.colorbar(im)


    return col_xs, row_ys, corners


def get_ROI(col_x, row_y, ROI_radius, n):
    '''
    Inputs:
    * col_x, row_y: the (x,y) pixel-position of the center of the ROI (dont have to be ints) \n
    *Returns*: \n
    * col_xs: the left and right column/x indices of the ROI
    * row_ys: the up and down row/y indices of the ROI
    '''

    d = int(ROI_radius)

    # the x,y indices delimiting the ROI
    # 06-16: added int()
    col_xs = [int(max(0, col_x-d)), int(min(n-1, col_x+d))]
    row_ys = [int(max(0, row_y-d)), int(min(n-1, row_y+d))]

    return col_xs, row_ys


def get_ROI_2(col_x, row_y, ROI_width, n):
    '''
    Inputs:
    * col_x, row_y: the (x,y) pixel-position of the center of the ROI (dont have to be ints) \n
    * ROI_width: in pixels. If odd, (x,y) will be centered. If even, (x,y) will be offset to bottom right. \n
    *Returns*: \n
    * col_xs: the left and right column/x indices of the ROI
    * row_ys: the up and down row/y indices of the ROI
    '''

    col_x = int(col_x)
    row_y = int(row_y)
    ROI_width = int(ROI_width)

    if ROI_width % 2: # is odd
        d1 = ROI_width//2
        d2 = d1
    else: # is even
        d1 = ROI_width/2
        d2 = d1 - 1

    # keep the int lol
    col_xs = [int(max(0, col_x-d1)), int(min(n-1, col_x+d2))]
    row_ys = [int(max(0, row_y-d1)), int(min(n-1, row_y+d2))]

    return col_xs, row_ys

def get_ROI_3(col_x, row_y, ROI_width, n):
    '''
    Inputs:
    * col_x, row_y: the (x,y) pixel-position of the center of the ROI (dont have to be ints) \n
    * ROI_width: in pixels. If odd, (x,y) will be centered. If even, (x,y) will be offset to bottom right. \n
    *Returns*: \n
    * lims_xy: (2, 2)-array [col_xs, row_ys]
        * col_xs: the left and right column/x indices of the ROI
        * row_ys: the up and down row/y indices of the ROI
    '''

    col_x = int(col_x)
    row_y = int(row_y)
    ROI_width = int(ROI_width)

    if ROI_width % 2: # is odd
        d1 = ROI_width//2
        d2 = d1
    else: # is even
        d1 = ROI_width/2
        d2 = d1 - 1

    lims_xy = np.zeros((2, 2), dtype=int)

    lims_xy[0, 0] = max(0, col_x-d1)
    lims_xy[0, 1] = min(n-1, col_x+d2)
    lims_xy[1, 0] = max(0, row_y-d1)
    lims_xy[1, 1] = min(n-1, row_y+d2)

    return lims_xy

def fit_ROI(f, image_arr, col_x, row_y, ROI_radius, show=True, **kwargs):
    '''
    TODO add option to input custom guesses \n
    *Inputs:* \n
    * f: the fit function as used in opt.curve_fit. For gaussian fitting, use gauss2d_gen_xy
    * image_arr: 2D square array of image data \n
    * col_x, row_y: the column index (x-coord) and row index (y-coord) which will be the center of the region where the fit will be performed
    * ROI_radius: the radius in pixels of the square region centered at (col_x, row_y), inside of which the fit will be performed \n
    *Returns:* \n
    * popt: the fitted parameters. e.g. (x0, y0, w, A, B) for f=gauss2d_gen_xy
    * ROI_bounds: the output of get_ROI(array([col_x, row_y]),...), useful for plotting
    '''

    n = image_arr.shape[0]     
    image_vec = np.ravel(image_arr)
    
    ### Selecting ROI
    col_xs, row_ys, _ = select_ROI(col_x, row_y, ROI_radius, n)

    # Cropping image -> ROI
    cropped_im = image_arr[row_ys[0]:row_ys[1], 
                           col_xs[0]:col_xs[1]]

    x_im = np.arange(0, n)
    y_im = np.arange(0, n)
    x_crop = x_im[col_xs[0]:col_xs[1]]
    y_crop = y_im[row_ys[0]:row_ys[1]]

    X,Y = np.meshgrid(x_crop, y_crop)
    X_vec = np.ravel(X)
    Y_vec = np.ravel(Y)

    ### The Fit
    xy_data = np.vstack((X_vec, Y_vec))
    z_data = np.ravel(cropped_im)

    guess = (
        col_x, # x0
        row_y, # y0
        0.5*ROI_radius, # w # for real data, modif this.
        image_vec.max(), # A
        0 # B
    )
    # print(x_crop) ## what index error?...
    my_bounds = np.array([
        (x_crop[0], x_crop[-1]), # x0
        (y_crop[0], y_crop[-1]), # y0
        (0, np.inf), # w
        (0, np.inf), # A
        (0, image_vec.max()) # B
    ])
    
    popt, pcov = opt.curve_fit(f, xy_data, z_data, p0 = guess, bounds=my_bounds.T) # seems to work as array rather than tuple

    if show:
        _ = plot_fit_3d_v2(image_arr, row_ys, col_xs, popt)

    return popt, (col_xs, row_ys)


def fit_ROI_2(f, im, col_x, row_y, ROI_width, show=True, **kwargs):
    '''
    In bolp5.4.2.1
    - verified that it works the same as og

    ### For rectangular images... n is the y/row-height?

    \nChanges v2:
    - modify guesses
    \nChanges v1:
    - variable names
    - guesses stay the same
    '''

    n = im.shape[0] # y/row-height
    
    # Selecting ROI
    lims = get_ROI_3(col_x, row_y, ROI_width, n)

    # Cropping image -> ROI
    cropped_im = im[slice2d(lims)]

    x_arr = np.arange(*lims[0]) # + 0.5
    y_arr = np.arange(*lims[1]) # + 0.5

    X,Y = np.meshgrid(x_arr, y_arr)
    X_vec = np.ravel(X)
    Y_vec = np.ravel(Y)

    ### The Fit
    xy_data = np.vstack((X_vec, Y_vec))
    z_data = np.ravel(cropped_im)

    guess = (
        col_x, # x0
        row_y, # y0
        0.25*ROI_width, # w # for real data, modif this.
        im.max(), # A
        0 # B
    )
    # print(x_crop) ## what index error?...
    my_bounds = np.array([
        (lims[0,0], lims[0,1]), # x0
        (lims[1,0], lims[1,1]), # y0
        (0, np.inf), # w
        (0, np.inf), # A
        (0, im.max()) # B
    ])
    
    popt, pcov = opt.curve_fit(f, xy_data, z_data, p0 = guess, bounds=my_bounds.T) # seems to work as array rather than tuple
    perr = np.sqrt(pcov.diagonal())

    if show:
        _ = plot_fit_3d_v3(f, im, lims[1], lims[0], popt)

    return popt, perr


# added 2025-08-17 when working on code-eIF4E
def fit_ROI_v3(f, im, col_x, row_y, ROI_width, show=True, **kwargs):
    '''
    - return: popt, perr
    v3: 
    - ok so og_2 already works with hilo yay. 
    - changed guesses for A,B
    - left bounds as is

    In bolp5.4.2.1
    - verified that it works the same as og

    ### For rectangular images... n is the y/row-height?
    
    \nChanges v2:
    - modify guesses
    \nChanges v1:
    - variable names
    - guesses stay the same
    '''

    n = im.shape[0] # y/row-height
    
    # Selecting ROI
    lims = get_ROI_3(col_x, row_y, ROI_width, n)
    # Cropping image -> ROI
    cropped_im = im[slice2d(lims)]

    x_arr = np.arange(*lims[0]) # + 0.5
    y_arr = np.arange(*lims[1]) # + 0.5
    X,Y = np.meshgrid(x_arr, y_arr)
    X_vec = np.ravel(X)
    Y_vec = np.ravel(Y)

    ### The Fit
    xy_data = np.vstack((X_vec, Y_vec))
    z_data = np.ravel(cropped_im)
    
    guess = (
        col_x, # x0
        row_y, # y0
        0.25*ROI_width, # w # for real data, modif this.
        im.max() - im.min(), # A
        im.mean() # B
    )
    # print(x_crop) ## what index error?...
    my_bounds = np.array([
        (lims[0,0], lims[0,1]), # x0
        (lims[1,0], lims[1,1]), # y0
        (0, np.inf), # w
        (0, np.inf), # A
        (im.min(), im.max()) # B
    ])
    
    popt, pcov = opt.curve_fit(f, xy_data, z_data, p0 = guess, bounds=my_bounds.T) # seems to work as array rather than tuple
    perr = np.sqrt(pcov.diagonal())

    if show:
        _ = plot_fit_3d_v3(f, im, lims[1], lims[0], popt)

    return popt, perr





def fit_1_track(f, im_stack, px_xy, ROI_width, silence=False):
    ''' 
    _Inputs:_
    * f: function as used by opt.curve_fit
    * im_stack: (N_step, n, n)-array
    * px_xy: initial (x,y) pixel position of the ptcl in first frame of im_stack \n
    * ROI_width
    \n_Returns:_
    * fit_P: (N_steps, 2*n_arg)-array, where n_arg = (# parameters of f) - 1
        - includes popt, perr for each step
    * stop_idx: int, last valid index of fit_P
    '''
    N_steps = im_stack.shape[0]
    n_arg = len(inspect.signature(f).parameters) - 1
    fit_P = np.zeros((N_steps, 2*n_arg))
    curr_xy = px_xy
    stop_idx = N_steps

    for i_step in range(N_steps):
        if not silence:
            print(f'Step {i_step}', flush=True)
        try:
            popt, perr = fit_ROI_v3(f, im_stack[i_step], *curr_xy, ROI_width, show=False)

            curr_xy = popt[0:2] # update next search position

            fit_P[i_step][0:n_arg] = popt
            fit_P[i_step][n_arg:2*n_arg] = perr

        except: # failed fit
            print(f'Failed step {i_step}')
            stop_idx = i_step
            break
    
    return fit_P, stop_idx




def fit_multi_3(f, im_stack, px_arr, ROI_radius, show_i=-1):
    ''' 
    Handle failed fits / disappearing ptcls
    _Inputs:_
    * f: function as used by opt.curve_fit
    * im_stack: (N_step, n, n)-array
    * px_arr: (N_ptcl, 2)-array of the (x, y) initial pixel-positions of the particles in first frame of im_stack \n
    _Returns:_
    * fit_Params: (N_steps, N_ptcl, n_arg)-array, where n_arg = (# parameters of f) - 1
    '''
    # n = im_stack.shape[1]
    N_ptcl = px_arr.shape[0]
    N_steps = im_stack.shape[0]
    n_arg = len(inspect.signature(f).parameters) - 1
    fit_Params = np.zeros((N_steps, N_ptcl, n_arg))

    curr_px = px_arr
    stop_idx = np.zeros(N_ptcl, dtype=int) + N_steps # last valid idx of i_ptcl is stop_idx[i_ptcl]-1. for plotting tracks: xy_arr[0:stop_idx[i_ptcl]]
    ptcl_to_skip = []

    for i_step in range(N_steps):
        print('Step', i_step, flush=True)
        for i_ptcl in range(N_ptcl):

            if i_ptcl not in ptcl_to_skip: # maybe not most efficient but easiest rn
                try:
                    popt, _ = fit_ROI(f, im_stack[i_step], curr_px[i_ptcl, 0], curr_px[i_ptcl, 1], ROI_radius, show=(i_ptcl==show_i)) # print for first ptcl
                    fit_Params[i_step][i_ptcl] = popt

                except: # failed fit
                    print(i_step, i_ptcl)
                    ptcl_to_skip.append(i_ptcl)
                    stop_idx[i_ptcl] = i_step
        
        # Set the next search spot
        curr_px = fit_Params[i_step, :, 0:2] # [previous step, all ptcls, first two params]
    
    return fit_Params, stop_idx

def plot_tracks_3(ax, fit_Params, stop_idx, line_color = 'spring', **kwargs):
    '''
    line_color (str): can be colormap or single color
     taken from colormap. e.g. Wistia, plasma, spring
    '''
    N_ptcl = fit_Params.shape[1]

    # Colormap for lines
    the_cmap = mpl.colormaps.get(line_color)
    if the_cmap != None:
        color_arr = the_cmap.resampled(N_ptcl)(range(N_ptcl))
    else:
        mpl.colors.to_rgba(line_color) # this will raise error for me
        color_arr = np.array([line_color]*N_ptcl)

    # Setting some default kwargs
    if 'lw' not in kwargs:
        kwargs['lw'] = 0.5
    # if ('c' not in kwargs) and ('color' not in kwargs):
    #     kwargs['color'] = 'yellow'

    for i_ptcl in range(N_ptcl):
        track = fit_Params[0:stop_idx[i_ptcl], i_ptcl, 0:2] # efficiency of array indexing like this?
        line = Line2D(track.T[0], track.T[1], color = color_arr[i_ptcl], **kwargs) 
        ax.add_line(line)
        line.set_label(f'ptcl {i_ptcl}')
    
    # set label for the last one
    # line.set_label('Fitted tracks')
    # f'Track {i_ptcl}') # maybe dont really care about the numbering in real data (but for sim..)



def track_lims(track, offset=5):
    '''
    // prob wanna change name to get_track_lims 
    - track: (N_ptcl, 2)-array
    '''

# offset = 5
    x,y = track.T

    # xmin = x.min()-5
    # xma

    lims = np.zeros((2,2), dtype=int) # why not
    for i in range(2):
        lims[i, 0] = track.T[i].min() - offset
        lims[i, 1] = track.T[i].max() + offset

    return lims

def get_track_lims(track, offset=5):
    ''' 
    - track: (N_ptcl, 2)-array
    '''

    # offset = 5
    x,y = track.T

    # xmin = x.min()-5
    # xma

    t_lims = np.zeros((2,2), dtype=int) # why not
    for i in range(2):
        t_lims[i, 0] = track.T[i].min() - offset
        t_lims[i, 1] = track.T[i].max() + offset

    return t_lims

def delay_track(track, m):
    '''
    Select an "m-delayed" subtrack of the input track, which contains every m-th position in track \n
    * track: expect a (N, 2)-array \n
    * m: 1 <= m <= N-1
    '''
    N = track.shape[0]
    if m not in range(1, N):
        raise ValueError('require 1 <= m <= N-1')  
    mask_arr = (np.arange(0, N) % m) == 0
    return track[mask_arr]

def track_dist(track):
    N = track.shape[0]
    displ = track[1:N] - track[0:N-1]
    SD = displ.T[0]**2 + displ.T[1]**2
    return np.sqrt(SD)

def MSD_lag(track, m):
    ''' 
    * track: (N, 2)-array of x,y-positions
    * m should range from from 0 to N-1
    '''
    N = track.shape[0]
    if m not in range(1, N):
        raise ValueError('require 1 <= m <= N-1')  
    delayed_track = delay_track(track, m)
    M = delayed_track.shape[0] 
    displ = delayed_track[1:M] - delayed_track[0:M-1]
    SD = displ.T[0]**2 + displ.T[1]**2
    return np.mean(SD)

def get_all_MSD(fit_Params, scale=None, dt=None):
    ''' 
    ***TODO***: need to modify for varying track sizes (stop_idx) \n
    *Inputs*:
    * fit_Params
    * to implement: scale, dt \n
    *Returns*:
    * msd_Arr: (N_ptcl, N_steps)-array of delayed MSDs, with delays m ranging from 0 to N_steps-1
    '''
    
    N_steps, N_ptcl = fit_Params.shape[0:2]
    msd_Arr = np.zeros((N_ptcl, N_steps-1))
    lag_arr = np.arange(1, N_steps)

    for i_ptcl in range(0, N_ptcl):
        track = fit_Params[:, i_ptcl, 0:2]
        for m in lag_arr:
            msd_Arr[i_ptcl][m-1] = MSD_lag(track, m)

    return msd_Arr



def get_all_MSD_2(fit_Params, scale=None, dt=None):
    ''' 
    ***TODO***: need to modify for varying track sizes (stop_idx) \n
    *Inputs*:
    * fit_Params
    * to implement: scale, dt \n
    *Returns*:
    * msd_Arr: (N_ptcl, N_steps)-array of delayed MSDs, with delays m ranging from 0 to N_steps-1
    '''
    
    N_steps, N_ptcl = fit_Params.shape[0:2]
    msd_Arr = np.zeros((N_ptcl, N_steps-1))
    lag_arr = np.arange(1, N_steps)
    xy_Arr = px2pos(fit_Params[:, :, 0:2])

    for i_ptcl in range(0, N_ptcl):
        track = xy_Arr[:, i_ptcl]
        for m in lag_arr:
            msd_Arr[i_ptcl][m-1] = MSD_lag(track, m)

    return msd_Arr


def get_all_MSD_3(fit_Params, stop_idx, scale=None, dt=None):
    ''' 
    ***TODO***: need to modify for varying track sizes (stop_idx) \n
    *Inputs*:
    * fit_Params
    * to implement: scale, dt \n
    *Returns*:
    * msd_Arr: (N_ptcl, N_steps)-array of delayed MSDs, with delays m ranging from 0 to N_steps-1
    '''
    
    N_steps, N_ptcl = fit_Params.shape[0:2]
    msd_Arr = np.zeros((N_ptcl, N_steps-1))
    lag_arr = np.arange(1, N_steps)
    xy_Arr = px2pos(fit_Params[:, :, 0:2])

    for i_ptcl in range(0, N_ptcl):
        track = xy_Arr[:stop_idx[i_ptcl], i_ptcl]
        for m in lag_arr[:stop_idx[i_ptcl]]:
            msd_Arr[i_ptcl][m-1] = MSD_lag(track, m)

    return msd_Arr


def MSD_lag_v2(track, m):
    ''' 
    * track: (N, 2)-array of x,y-positions
    * m should range from from 0 to N-1
    '''
    N = track.shape[0]
    if m not in range(1, N):
        raise ValueError('require 1 <= m <= N-1')  
    delayed_track = delay_track(track, m)
    M = delayed_track.shape[0] 
    displ = delayed_track[1:M] - delayed_track[0:M-1]
    SD = displ.T[0]**2 + displ.T[1]**2
    return np.mean(SD)


def MSD_curve_2(fp, stop, i_ptcl):
    ''' 
    - returns t_arr, msd_curve
    '''
    track_px = fp[:stop[i_ptcl], i_ptcl, 0:2]
    track_pos = px2pos(track_px)
    # N_steps = fp.shape[0]

    msd_curve = np.zeros(stop[i_ptcl]-1)
    lag_arr = np.arange(1, stop[i_ptcl])

    for m in lag_arr:
        msd_curve[m-1] = MSD_lag_v2(track_pos, m)

    t_arr = lag_arr*ExpInfo.dt*10**-3

    return t_arr, msd_curve

def MSD_curve_3(fp, stop_i, show=True):
    ''' 
    - return t_arr (simply lag_arr in units of t), msd_curve
    '''
    track_px = fp[:stop_i, 0:2]
    track_pos = px2pos(track_px)
    N_steps = fp.shape[0]

    msd_curve = np.zeros(stop_i-1)
    lag_arr = np.arange(1, stop_i)

    for m in lag_arr:
        msd_curve[m-1] = MSD_lag_v2(track_pos, m)

    t_arr = lag_arr*ExpInfo.dt*10**-3

    if show:
        fig, ax = plt.subplots()
        ax.plot(t_arr, msd_curve, c = 'black', lw=1)
        ax.set(xlabel = f'Lag time [s]', ylabel = f'MSD [${ExpInfo.unit}^2$]')

    return t_arr, msd_curve

def linear(x, a, b):
    return a*x + b

def MSD_curve_ij(fp, stop_i, i=2, j=5, show=True):
    '''
    - yes stop_i is important bcz dont wanna consider bad pts in the lag calc
    '''
    track_px = fp[:stop_i, 0:2]
    track_pos = px2pos(track_px)

    lag_arr = np.arange(i, j+1)
    n_pts = j+1-i
    msd_curve = np.zeros(n_pts)

    for idx in range(n_pts):
        m = lag_arr[idx]
        msd_curve[idx] = MSD_lag(track_pos, m)

    t_arr = lag_arr*ExpInfo.dt*10**-3

    ## Fit

    popt, pcov = opt.curve_fit(linear, t_arr, msd_curve)

    D = popt[0]/4

    if show:
        fig, ax = plt.subplots()
        ax.scatter(t_arr, msd_curve, c='black')
        ax.plot(t_arr, linear(t_arr, *popt), c='black', ls='--')
        ax.set_xticks(t_arr)
        ax.set_xticklabels(1000*t_arr)
        ax.set(xlabel = f'Lag time [{ExpInfo.dt_unit}]', ylabel = f'MSD [${ExpInfo.unit}^2$]')

        ax.set_title(f'D ({i} to {j}) = {round(D, 4)} ${ExpInfo.unit}^2/s$')
        display(Math(f'D ({i} to {j}) = {round(D, 4)} {ExpInfo.unit}^2/s'))


    return D


''' there's an error with this one

def plot_MSD_2(fp, stop):
    N_steps = fp.shape[0]
    N_ptcl = fp.shape[1]

    plt.style.use('seaborn-v0_8-darkgrid')

    fig, ax = plt.subplots()


    msd_Arr = get_all_MSD_3(fp, stop)
    lag_arr = np.arange(1, N_steps)*ExpInfo.dt*10**-3


    # colors
    the_cmap = mpl.colormaps.get('spring')
    color_arr = the_cmap.resampled(N_ptcl)(range(N_ptcl))


    for i_ptcl in range(N_ptcl):
        ax.plot(lag_arr[:stop[i_ptcl]], msd_Arr[i_ptcl], label = f'{i_ptcl}', c = color_arr[i_ptcl], lw=2)

    # ax.set(xlabel = f'Lag time ({ExpInfo.dt_unit})', ylabel = f'MSD [{ExpInfo.unit}$^2$]')
    ax.set(xlabel = f'Lag time [s]', ylabel = f'MSD [{ExpInfo.unit}$^2$]')
    ax.legend()
    plt.style.use('default')
    return fig, ax
'''

'''
WT
'''


def wt_stack(im, n_kern, a_min, a_max, a_step, **kwargs):
    ''' 
    Returns:
    * WT_arr, (X, Y, Z_arr), a_arr
    '''


    n = im.shape[0]
    
    # n_kern = 128
    n_arr = np.arange(-n_kern//2, n_kern//2, 1)
    X, Y = np.meshgrid(n_arr, n_arr)

    # Z_arr = np.zeros(())


    # a_step = 0.5
    # a_min = 1
    # a_max = n_kern/2
    a_arr = np.arange(a_min, a_max, a_step)

    Z_arr = np.zeros((a_arr.size, n_kern, n_kern))
    WT_arr = np.zeros((a_arr.size, n, n))


    #a = 10
    for i in range(a_arr.size):
        print('Convol', i, flush=True)
        a = a_arr[i]
        Z_arr[i] = ricker2d(X, Y, a)
        # Z_arr[i] = Z_rick
        WT_arr[i] = sig.convolve2d(im, Z_arr[i], mode='same', **kwargs) #, boundary='wrap')

    return WT_arr, (X, Y, Z_arr), a_arr


def wt_a(im, n_kern, a, **kwargs):
    print(1, flush=True)

    n_arr = np.arange(-n_kern//2, n_kern//2, 1)
    X, Y = np.meshgrid(n_arr, n_arr)

    Z = ricker2d(X, Y, a)
    WT_arr = sig.convolve2d(im, Z, mode='same', **kwargs) #, boundary='wrap')

    return WT_arr, (X, Y, Z)

def wt_a_v2(im, n_kern, a, **kwargs):
    n_arr = np.arange(-n_kern//2, n_kern//2, 1)
    X, Y = np.meshgrid(n_arr, n_arr)

    Z = ricker2d(X, Y, a)
    WT_arr = sig.convolve2d(im, Z, **default_kwargs(kwargs, mode='same', boundary='wrap')) 
    return WT_arr