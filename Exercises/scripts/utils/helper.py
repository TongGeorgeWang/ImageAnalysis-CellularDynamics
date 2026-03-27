import numpy as np
import scipy.signal
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import inspect
import itertools

from IPython.display import HTML, display


# 2025-06-25: switched interpreter: ~\AppData\...\python.exe -> 'base' Conda ~\anaconda3\python.exe (both are Python 3.11.4)



'''
General
'''
def slice2d(lims):
    ''' 
    * lims: (2,2)-arr
    '''
    return (slice(*lims[1]), slice(*lims[0]))


def default_kwargs(input_kwargs, **defaults):
    '''
    The input kwargs override the default kwargs. If something is missing from the input, it is covered by default.
    '''
    defaults.update(input_kwargs)
    return defaults


'''
Image functinos
'''


class ExpInfo:
    '''
    _Fields_:
    * scale: the length of 1 pixel in the image (e.g. for simulation: 10**-7 m) \n
    * dt: frame rate
    '''
    # these would be defaults if only working with one Experiment
    scale = None # 10**-7
    unit = None # 'm'
    n = None
    dt = None
    dt_unit = None

    # def __init__(self, scale, unit, dt, dt_unit, name)



class CycleFunc:

    def __init__(self, func_i, num, *args):
        ''' 
        - num is number of iterations, e.g. # of frames. i.e. num = "max valid index"+1
        '''
        self.func_i = lambda i: func_i(i, *args)
        self.num = int(num)
        self.i = -1
        self.args = args
    
    def move(self, x, silent = False):
        self.i += int(x)
        self.i = self.i % self.num
        if not silent:
            print(self.i)
        return self.func_i(self.i)
    
    def set_i(self, i, show = True):
        self.i = int(i) % self.num
        if show:
            print(self.i)
            return self.func_i(self.i)


# 2025-06-19 temp still using helper.py
# SCALE = 10**-7

def pos2px(pos_arr):
    return (pos_arr//ExpInfo.scale).astype(int)

def swap_cols(arr, i, j):
    arr[:, [i, j]] = arr[:, [j, i]]

def px2idx(px_arr):
    idx_arr = np.copy(px_arr)
    swap_cols(idx_arr, 0, 1)
    return idx_arr

def px2pos(px_arr):
    return px_arr*ExpInfo.scale


def make_image(pos_arr, n, Z_gauss, silence=False):
    '''
    Creates convolved image from particle positions. \n 
    _Inputs:_
    * pos_arr: (N_ptcl, 2)-array containing real xy-positions in meters
    * n: to specify the size of the image, an (n, n)-array
    * Z_gauss: the output of gauss_kern
    * silence: whether or not to print the number of particles that did not move off the FOV \n
    _Returns:_
    * conv_im: the convolved image
    '''
    image = np.zeros((n, n))
    px_arr = pos2px(pos_arr)
    idx_arr = px2idx(px_arr)
    c = 0
    for (row, col) in idx_arr:
        if (row >= 0 and row < n and col >= 0 and col < n):
            image[row, col] = 1
            c += 1
    if not silence:
        print(c)
    conv_im = scipy.signal.convolve2d(image, Z_gauss, mode='same') # same sized output...
    return conv_im


def simul_imag(N_ptcl = 1, n = 256, PSF_param = (6, 8), noise_param = (1/0.2, 100), cmap = 'gray', show = True):
    '''
    Creates an image with N_ptcl placed randomly, with noise. \n
    PSF_param = (e^{-2} width w in px, radius of convolution kernel in px) \n
    noise_param = (SNR, N_photons)
    '''

    w, n_kern = PSF_param
    SNR, N_photons = noise_param
    s_level = 1+1/(SNR-1)

    ### CREATE SIMULATION
    pos_arr = initial_pos(n, N_ptcl)
    Z_gauss = gauss_kern_2(w, n_kern, show=show)
    conv_im = make_image(pos_arr, n, Z_gauss)

    # add noise
    conv_im += 1/(SNR-1)
    noisy_im = np.random.poisson(conv_im*N_photons)/N_photons

    if show:
        fig = plt.figure()
        # fig.set_figwidth(fig.get_figwidth()*2)
        ax = fig.add_subplot()
        im = ax.imshow(noisy_im, cmap=cmap, interpolation='none', vmin=0, vmax=s_level) #1)
        plt.colorbar(im)

    return noisy_im


def simul_imag_2(N_ptcl = 1, n = 256, PSF_param = (6, 8), noise_param = (1/0.2, 100), cmap = 'gray', show = [0, 1, 2]):
    '''
    Simul_imag_2: same as simul_imag but can display noise and signal separately \n
    PSF_param = (e^{-2} width w in px, radius of convolution kernel in px) \n
    noise_param = (SNR, N_photons) \n
    show: list, 0 for noisy_im, 1 for signal_arr, 2 for noise_arr, empty [] for no show \n
    returns: ims, (fig, ax)
    '''

    w, n_kern = PSF_param
    SNR, N_photons = noise_param
    s_level = 1+1/(SNR-1)

    ### CREATE SIMULATION
    pos_arr = initial_pos(n, N_ptcl)
    Z_gauss = gauss_kern_2(w, n_kern, show=show)
    signal_arr = make_image(pos_arr, n, Z_gauss)

    # add noise
    # signal_arr += 1/(SNR-1) -> mistake? would be adding twice..
    noise_arr = np.zeros((n, n)) + 1/(SNR-1)

    signal_arr = np.random.poisson(signal_arr*N_photons)/N_photons
    noise_arr = np.random.poisson(noise_arr*N_photons)/N_photons

    noisy_im = signal_arr + noise_arr

    ims = (noisy_im, signal_arr, noise_arr)
    titles = ('Signal+Noise', 'Signal', 'Noise')

    if len(show) > 0:
        fig, ax = plt.subplots(1, len(show))
        if len(show) == 1:
            ax = [ax] # :)
        fig.set_figwidth(fig.get_figwidth()*len(show))
        for i in show:
            im = ax[i].imshow(ims[i], cmap=cmap, interpolation='none', vmin=0, vmax=s_level) #1)
            ax[i].set(title = titles[i])
            plt.colorbar(im)

    return ims, (fig, ax)

def simul_imag_pos(pos_arr, n = 256, PSF_param = (6, 8), noise_param = (1/0.2, 100), cmap = 'gray', show = True, show_kern = True):
    '''
    PSF_param = (e^{-2} width w in px, radius of convolution kernel in px) \n
    noise_param = (SNR, N_photons) \n
    // no: would have to change make_ani: show: [gauss_kern, imshow] \n
    returns: noisy_im
    '''

    w, n_kern = PSF_param
    SNR, N_photons = noise_param
    s_level = 1+1/(SNR-1)

    ### CREATE SIMULATION
    Z_gauss = gauss_kern_2(w, n_kern, show=(show and show_kern))
    conv_im = make_image(pos_arr, n, Z_gauss, silence=True)

    # add noise
    conv_im += 1/(SNR-1)
    noisy_im = np.random.poisson(conv_im*N_photons)/N_photons

    if show:
        fig = plt.figure()
        # fig.set_figwidth(fig.get_figwidth()*2)
        ax = fig.add_subplot()
        im = ax.imshow(noisy_im, cmap=cmap, interpolation='none', vmin=0, vmax=s_level) #1)
        plt.colorbar(im)

    return noisy_im



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

def get_val_range_tmp(SNR):
    s_level = 1+1/(SNR-1)
    value_range = {'vmin':0,
                   'vmax':s_level}
    return value_range


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
    im = ax.imshow(cropped_im, extent=(col_xs[0], col_xs[1], row_ys[1], row_ys[0]))
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


'''
Diffusion simulation
'''

D_coeff = 0.5*10**-12 # m^2/s
dt_rw = 50*10**-3 # s
a_rw = np.sqrt(4*D_coeff*dt_rw) # random walk step size


def initial_pos(n, N_ptcl):
    '''
    **Gotta change name! get_initial_pos** \n
    Randomly chooses N_ptcl out of n^2. \n
    *Returns:* \n
    - pos_arr: (N_ptcl, 2)-array, pos_arr[i] = [x_i, y_i] \n
    '''
    N_px = n**2
    # Random initial positions
    idx_choice = np.random.choice(np.arange(N_px), N_ptcl)
    pos_arr = np.array((idx_choice // n, idx_choice % n)).T * 10**-7 # ? nm = ? px * 100 nm  
    return pos_arr


def update_pos(pos_arr, a_rw, new_arr=False):
    if new_arr:
        pos_arr = np.copy(pos_arr)
    # else will dynamically modify pos_arr
    N_ptcl = pos_arr.shape[0]
    th_arr = 2*np.pi*np.random.random(N_ptcl)
    for i in range(N_ptcl):
        th = th_arr[i]
        x = a_rw*np.cos(th)
        y = a_rw*np.sin(th)
        pos_arr[i][0] += x
        pos_arr[i][1] += y
    
    return pos_arr



def simul_pos(initial_pos, N_steps, a_rw):
    '''
    *Returns:* \n
    - pos_Arr: (N_steps, N_ptcl, 2)-shaped array, containing the (N_ptcl, 2)-shaped position arrays at each of the N_steps steps
    '''
    N_ptcl = initial_pos.shape[0]
    pos_Arr = np.zeros((N_steps, N_ptcl, 2))
    curr_pos = initial_pos
    for i in range(N_steps):
        pos_Arr[i] = curr_pos
        update_pos(curr_pos, a_rw) # ah don't even need the new_arr=True since copying it in pos_Arr
    return pos_Arr


def simul_imstack(pos_Arr, n, PSF_param = (6, 8), noise_param = (1/0.2, 100)):
    '''
    Simulate image stack
    *Inputs:* \n
    * pos_Arr = the output of simul_pos(initial_pos, N_steps, a_rw) \n
    * show: for displaying PSF kernel.
    *Returns:* \n
    * image_Arr: (N_steps, n, n)-array \n
    '''
    image_Arr = np.zeros((pos_Arr.shape[0], n, n)) # pos_Arr.shape[0] is N_steps

    for i in range(pos_Arr.shape[0]):
        pos_arr = pos_Arr[i]
        noisy_im = simul_imag_pos(pos_arr, n, PSF_param, noise_param, show=False)
        np.copyto(image_Arr[i], noisy_im)
    
    return image_Arr



def show_ROI(ax, col_xs, row_ys, box_kwargs = {}, marker_kwargs = {}, show_corners = True):
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


'''
Animation
'''



def make_img_frames(fig, ax, im_stack, **kwargs):
    frames = []
    N_steps = im_stack.shape[0]
    
    for i in range(N_steps):
        im_arr = im_stack[i]
        img = ax.imshow(im_arr, animated = True, **kwargs)
        frames.append([img])
        # print(i)
    
    fig.colorbar(img, ax=ax)

    return frames


# future plan:
# def make_track_frames():
#     ...





def anim_im_stack(fig, ax, im_stack, show=True, **kwargs):
    frames = make_img_frames(fig, ax, im_stack, **kwargs)

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=True)

    html_anim = HTML(ani.to_jshtml())
    if show:
        display(html_anim)

    return ani, html_anim

def anim_im_stack_2(fig, ax, im_stack, **kwargs):
    frames = make_img_frames(fig, ax, im_stack, **kwargs)
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=True)
    return ani


def plot_tracks(ax, fit_Params, **kwargs):
    ''' 
    all same color
    '''
    N_ptcl = fit_Params.shape[1]

    # Setting some default kwargs
    if 'lw' not in kwargs:
        kwargs['lw'] = 0.5
    if ('c' not in kwargs) and ('color' not in kwargs):
        kwargs['color'] = 'yellow'

    for i_ptcl in range(N_ptcl):
        track = fit_Params[:, i_ptcl, 0:2] # efficiency of array indexing like this?
        line = Line2D(track.T[0], track.T[1], **kwargs) 
        ax.add_line(line)
    
    # set label for the last one
    line.set_label('Fitted tracks')
    # f'Track {i_ptcl}') # maybe dont really care about the numbering in real data (but for sim..)


# 06-25: theres an error actually. the else case of whether valid cmap inputted
def plot_tracks_cmap(ax, fit_Params, cmap_for_lines = 'spring', **kwargs):
    '''
    line colors taken from colormap. e.g. Wistia, plasma, spring
    '''
    N_ptcl = fit_Params.shape[1]

    # Colormap for lines
    the_cmap = mpl.colormaps.get(cmap_for_lines)
    if the_cmap != None:
        line_colors = the_cmap.resampled(N_ptcl)(range(N_ptcl))
    else:
        raise ValueError('not a colormap. TODO: handle this case for inputting colors')

    # Setting some default kwargs
    if 'lw' not in kwargs:
        kwargs['lw'] = 0.5
    # if ('c' not in kwargs) and ('color' not in kwargs):
    #     kwargs['color'] = 'yellow'

    for i_ptcl in range(N_ptcl):
        track = fit_Params[:, i_ptcl, 0:2] # efficiency of array indexing like this?
        line = Line2D(track.T[0], track.T[1], color = line_colors[i_ptcl], **kwargs) 
        ax.add_line(line)
    
    # set label for the last one
    line.set_label('Fitted tracks')
    # f'Track {i_ptcl}') # maybe dont really care about the numbering in real data (but for sim..)



'''
Math functions
'''
def gauss2d(x, y, w):
    return np.exp(-2*(x**2+y**2)/(w**2))

def ricker2d(x, y, a):
    return (1/a)*(2-((x/a)**2+(y/a)**2))*np.exp(-(x**2+y**2)/(2*a**2))

def gauss2d_xy(xy, w):
    x = xy[0]
    y = xy[1]
    return np.exp(-2*(x**2+y**2)/(w**2))


def gauss2d_gen(x, y, x0, y0, w, A, B):
    return A*np.exp(-2*((x-x0)**2+(y-y0)**2)/(w**2))+B

def gauss2d_gen_xy(xy, x0, y0, w, A, B):
    x = xy[0]
    y = xy[1]
    return A*np.exp(-2*((x-x0)**2+(y-y0)**2)/(w**2))+B

def gauss1d(x, w):
    return np.exp(-2*(x/w)**2)

def gauss1d_gen(x, w, x0, A, B):
    return A*np.exp(-2*((x-x0)/w)**2)+B


def ricker2d_gen(x, y, x0, y0, a, A, B):
    '''
    Comments:
    * only one scaling parameter a: takes care of both height and width at same time (and normilissation) 
    '''
    # return (1/a)*(2-((x-x0)/a)**2-((y-y0)/a)**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*a**2))
    return (A/a)*(2-((x-x0)/a)**2-((y-y0)/a)**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*a**2))+B

def ricker2d_gen_2(x, y, x0, y0, a, A, B):
    '''
    Comments:
    - this one has one the A as prefactor
    '''
    return A*(2-((x-x0)/a)**2-((y-y0)/a)**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*a**2))+B

def ricker2d_gen_xy(xy, x0, y0, a, A, B):
    '''
    Comments:
    * only one scaling parameter a: takes care of both height and width at same time (and normilissation) 
    '''
    x = xy[0]
    y = xy[1]
    return (A/a)*(2-((x-x0)/a)**2-((y-y0)/a)**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*a**2))+B


def ricker2d_gen_xy_2(xy, x0, y0, a, A, B):
    '''
    2025-08-17:
    - as of rn i think this should be the preferred one. the "normalization" with 1/a actually doesnt matter at all
    - bcz you are adjusting the height anyways to match the height of the signal. 
    - there's no such thing as "normalization" in this context.
    - no matter what: the fitted wavelet will have height = the signal height. 
    - if you enforce a scaling of 1/a, well the fitted A will just compensate to achieve the signal height.

    \nComments:
    * only one scaling parameter a: takes care of both height and width at same time (and normilissation) 
    '''
    x = xy[0]
    y = xy[1]
    return A*(2-((x-x0)/a)**2-((y-y0)/a)**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*a**2))+B


'''
Plotting
'''

def gauss_kern(w, n_kern, show=False):

    x = np.arange(-n_kern, n_kern+1, 1)
    y = np.arange(-n_kern, n_kern+1, 1)
    X, Y = np.meshgrid(x, y)
    Z_gauss = gauss2d(X, Y, w)

    if show:
        fig = plt.figure()
        fig.set(size_inches=fig.get_size_inches()*0.75)
        ax = fig.add_subplot(projection='3d')
        # ax.plot_surface(X, Y, Z_gauss)
        ax.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z_gauss))
        plt.show()

    return X, Y, Z_gauss

def gauss_kern_show(w, n_kern):

    x = np.arange(-n_kern, n_kern+1, 1)
    y = np.arange(-n_kern, n_kern+1, 1)
    X, Y = np.meshgrid(x, y)
    Z_gauss = gauss2d(X, Y, w)

    fig = plt.figure()
    fig.set(size_inches=fig.get_size_inches()*0.75)
    ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(X, Y, Z_gauss)
    ax.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z_gauss))
    plt.show()

    return Z_gauss


def gauss_kern_2(w, n_kern, show=False):

    x = np.arange(-n_kern, n_kern+1, 1)
    y = np.arange(-n_kern, n_kern+1, 1)
    X, Y = np.meshgrid(x, y)
    Z_gauss = gauss2d(X, Y, w)

    if show:
        fig = plt.figure()
        fig.set(size_inches=fig.get_size_inches()*0.75)
        ax = fig.add_subplot(projection='3d')
        # ax.plot_surface(X, Y, Z_gauss)
        ax.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z_gauss))
        ax.set(title = f'PSF kernel (w = {w})')
        plt.show()

    return Z_gauss




def alpha_scatter(image_arr, x_im, y_im, fig_ax = None, size_option = 'default', rgb=(0.8, 0, 0.8)):
    '''
    x_im, y_im are the 1D arrays e.g. 0 to n-1 (!! or the x_crop/y_crop) 'n
    fig_ax = None or tuple (fig, ax) \n 
    size_option = 'default', 'alpha' or the desired size (int)
    '''

    # I could do cases... based on size/shape of input params ye

    image_vec = np.ravel(image_arr)
    alpha_vec = image_vec/image_vec.max()
    # print(alpha_vec.max(), alpha_vec.min())
    rgba = np.zeros((alpha_vec.size, 4))
    rgba[:,0]+=rgb[0]
    rgba[:,1]+=rgb[1]
    rgba[:,2]+=rgb[2]
    rgba[:,3] = alpha_vec

    X_im, Y_im = np.meshgrid(x_im, y_im)

    '''
    n = image_arr.shape[0]
    # case: x, y are the base arrays e.g. 0 to n-1
    if x.size == n
    # case: x, y are the 2D meshgrids
    # case: x, y are the linearized meshgrids
    '''

    if fig_ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = fig_ax

    # match case ...
    if size_option == 'default':
        the_size = 20
    elif size_option == 'alpha':
        the_size = alpha_vec
    else:
        the_size = size_option

    ax.scatter(np.ravel(X_im), np.ravel(Y_im), np.ravel(image_arr), c=rgba, s=the_size) # s=alpha_vec, c = rgba) 
    # plt.show()
    return fig, ax


def alpha_scatter_v2(image_arr, x_im, y_im, fig_ax = None, size_option = 'default', rgb=(0.8, 0, 0.8)):
    '''
    i was gonna make one with the x_crop/y_crop figured out but oh well \n
    x_im, y_im are the 1D arrays e.g. 0 to n-1 \n
    fig_ax = None or tuple (fig, ax) \n
    size_option = 'default', 'alpha' or the desired size (int)
    '''

    # I could do cases... based on size/shape of input params ye

    image_vec = np.ravel(image_arr)
    alpha_vec = image_vec/image_vec.max()
    # print(alpha_vec.max(), alpha_vec.min())
    rgba = np.zeros((alpha_vec.size, 4))
    rgba[:,0]+=rgb[0]
    rgba[:,1]+=rgb[1]
    rgba[:,2]+=rgb[2]
    rgba[:,3] = alpha_vec

    X_im, Y_im = np.meshgrid(x_im, y_im)

    '''
    n = image_arr.shape[0]
    # case: x, y are the base arrays e.g. 0 to n-1
    if x.size == n
    # case: x, y are the 2D meshgrids
    # case: x, y are the linearized meshgrids
    '''

    if fig_ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = fig_ax

    # match case ...
    if size_option == 'default':
        the_size = 20
    elif size_option == 'alpha':
        the_size = alpha_vec
    else:
        the_size = size_option

    ax.scatter(np.ravel(X_im), np.ravel(Y_im), np.ravel(image_arr), c=rgba, s=the_size) # s=alpha_vec, c = rgba) 
    # plt.show()
    return fig, ax



def alpha_scatter_v1(image_arr, x_im, y_im, fig_ax = None, size_option = 'default', rgb=(0.8, 0, 0.8)):
    '''
    x_im, y_im are the 1D arrays e.g. 0 to n-1
    fig_ax = None or tuple (fig, ax)
    size_option = 'default', 'alpha' or the desired size (int)
    '''

    # I could do cases... based on size/shape of input params ye

    image_vec = np.ravel(image_arr)
    alpha_vec = image_vec/image_vec.max()
    # print(alpha_vec.max(), alpha_vec.min())
    rgba = np.zeros((alpha_vec.size, 4))
    rgba[:,0]+=rgb[0]
    rgba[:,1]+=rgb[1]
    rgba[:,2]+=rgb[2]
    rgba[:,3] = alpha_vec

    X_im, Y_im = np.meshgrid(x_im, y_im)

    if fig_ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = fig_ax

    # match case ...
    if size_option == 'default':
        the_size = 20
    elif size_option == 'alpha':
        the_size = alpha_vec
    else:
        the_size = size_option

    ax.scatter(np.ravel(X_im), np.ravel(Y_im), np.ravel(image_arr), c=rgba, s=the_size) # s=alpha_vec, c = rgba) 
    # plt.show()
    return fig, ax


# 2025-06-12
# was gonna simply rename but:
def plot_fit_3d_v3(f, image_arr, row_ys, col_xs, popt):
    ''' 
    - f needs to be in the _xy form
    '''

    cropped_im = image_arr[row_ys[0]:row_ys[1], col_xs[0]:col_xs[1]]
    n = image_arr.shape[0]
    x_im = np.arange(0, n)
    y_im = np.arange(0, n)
    x_crop = x_im[col_xs[0]:col_xs[1]]
    y_crop = x_im[row_ys[0]:row_ys[1]]

    fig, ax = alpha_scatter(cropped_im, x_crop, y_crop, rgb=(1, 0, 1))
    ax.set(xlabel='x', ylabel='y', title='fit')

    x_lin = np.linspace(x_crop[0], x_crop[-1], 50)
    y_lin = np.linspace(y_crop[0], y_crop[-1], 50)
    X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
    Z_fit = f((X_lin, Y_lin), *popt)

    ax.plot_surface(X_lin, Y_lin, Z_fit, color=(1, 0, 1, 0.2))

    return fig, ax


def plot_fit_3d_v2(image_arr, row_ys, col_xs, popt):

    cropped_im = image_arr[row_ys[0]:row_ys[1], col_xs[0]:col_xs[1]]
    n = image_arr.shape[0]
    x_im = np.arange(0, n)
    y_im = np.arange(0, n)
    x_crop = x_im[col_xs[0]:col_xs[1]]
    y_crop = x_im[row_ys[0]:row_ys[1]]

    fig, ax = alpha_scatter(cropped_im, x_crop, y_crop, rgb=(1, 0, 1))
    ax.set(xlabel='x', ylabel='y', title='fit')

    x_lin = np.linspace(x_crop[0], x_crop[-1], 50)
    y_lin = np.linspace(y_crop[0], y_crop[-1], 50)
    X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
    Z_fit = gauss2d_gen(X_lin, Y_lin, popt[0], popt[1], popt[2], popt[3], popt[4])

    ax.plot_surface(X_lin, Y_lin, Z_fit, color=(1, 0, 1, 0.2))

    return fig, ax

def plot_fit_3d(image_arr, cxs, cys, popt):

    cropped_im = image_arr[cxs[0]:cxs[1], cys[0]:cys[1]]
    n = image_arr.shape[0]
    x_im = np.arange(0, n)
    y_im = np.arange(0, n)
    x_crop = x_im[cxs[0]:cxs[1]]
    y_crop = x_im[cys[0]:cys[1]]

    fig, ax = alpha_scatter(cropped_im, x_crop, y_crop, rgb=(1, 0, 1))
    ax.set(xlabel='x', ylabel='y', title='fit')

    x_lin = np.linspace(x_crop[0], x_crop[-1], 50)
    y_lin = np.linspace(y_crop[0], y_crop[-1], 50)
    X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
    Z_fit = gauss2d_gen(X_lin, Y_lin, popt[0], popt[1], popt[2], popt[3], popt[4])

    ax.plot_surface(X_lin, Y_lin, Z_fit, color=(1, 0, 1, 0.2))

    return fig, ax

'''
Data analysis
'''



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



# 06-25: fixed inspect.signature(f) rather than ..(gauss2d_gen_xy)
def fit_multi(f, im_stack, px_arr, ROI_radius, show_i=-1):
    ''' 
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

    # xy_Arr = np.zeros((N_steps, N_ptcl, 2))

    curr_px = px_arr

    for step in range(N_steps):
        for i in range(N_ptcl):
            popt, _ = fit_ROI(f, im_stack[step], curr_px[i, 0], curr_px[i, 1], ROI_radius, show=(i==show_i)) # print for first ptcl
            fit_Params[step][i] = popt
            # xy_Arr[step][i] = popt[0:2]
            # print(popt[0:2])
        
        # Set the next search spot
        ## curr_px = xy_Arr[step].astype(int) ## BIG MISTAKE !! xy_Arr not even defined wth
        curr_px = fit_Params[step, :, 0:2] # [previous step, all ptcls, first two params]
    
    return fit_Params


def fit_1_max(f, image_arr, ROI_radius, show=True, **kwargs):
    ''' 
    For images containing one peak. 
    *Inputs:* \n
    * f: the fit function as used in opt.curve_fit. For gaussian fitting, use gauss2d_gen_xy
    * image_arr: 2D square array of image data \n
    * ROI_radius: the radius in pixels of the square region centered at the peak, inside of which the fit will be performed \n
    *Returns:* \n
    * popt: the fitted parameters. e.g. (x0, y0, w, A, B) for f=gauss2d_gen_xy
    * ROI_bounds: the output of get_ROI(array([col_x, row_y]),...), useful for plotting
    '''
    n = image_arr.shape[0]
    m = np.argmax(image_arr)
    col_x, row_y = (m % n, m // n)
    return fit_ROI(f, image_arr, col_x, row_y, ROI_radius, show, **kwargs)




def label_ptcl_2(ax, px_arr, **kwargs):
    for i in range(px_arr.shape[0]):
        (x, y) = px_arr[i]
        ax.scatter(x, y, **kwargs) #, lw=1)
        ax.annotate(str(i), (x, y), textcoords='offset points', xytext=(5,5), ha='center')


def round_digit(arr, n_digits):
    """
    (By Gemini) \n
    Rounds a NumPy array to a specified number of significant digits.

    Args:
        arr (np.ndarray): The input NumPy array.
        n_digits (int): The number of significant digits to round to.

    Returns:
        np.ndarray: The array rounded to the specified significant digits.
    """
    # Handle zeros separately to avoid log10(0)
    non_zero_mask = arr != 0
    result = np.zeros_like(arr, dtype=float)

    if np.any(non_zero_mask):
        # Calculate the exponent for non-zero elements
        # floor(log10(abs(x))) gives the power of 10 of the most significant digit.
        # Subtracting (n_digits - 1) makes sure we're rounding at the correct place.
        exponent = np.floor(np.log10(np.abs(arr[non_zero_mask]))) - (n_digits - 1)

        # Calculate the scaling factor
        scale = 10**exponent

        # Round and then scale back
        result[non_zero_mask] = np.round(arr[non_zero_mask] / scale) * scale

    return result

def check_track(fit_Params, px_Arr, i_ptcl):
    fit = fit_Params[:, i_ptcl, 0:2]
    w_fit = fit_Params[:, i_ptcl, 2]
    real = px_Arr[:, i_ptcl, :]
    fig, ax = plt.subplots()

    image_box = Line2D([0, 255, 255, 0, 0], [0, 0, 255, 255, 0], ls='--', color='black')
    image_box.set_label('Image boundary')
    ax.add_line(image_box)

    line = Line2D(fit.T[0], fit.T[1], lw=1, color='black') #'#00FFFF') #'#FFA500')
    line.set_label('Fit')
    ax.add_line(line)

    # ax.autoscale()

    ax.scatter(real[0,0], real[0,1], c='red', label='Actual')
    label_ptcl_2(ax, real, c='red')


    ax.legend()
    ax.set_title(f'ptcl {i_ptcl}')

    ### Axes limits
    xmin = min(real.T[0].min()-1, fit.T[0].min()-1)
    xmax = max(real.T[0].max()+1, fit.T[0].max()+1)
    ymin = min(real.T[1].min()-1, fit.T[1].min()-1)
    ymax = max(real.T[1].max()+1, fit.T[1].max()+1)
    # ax.set_xlim(real.T[0].min()-1, real.T[0].max()+1)
    # ax.set_ylim(real.T[1].min()-1, real.T[1].max()+1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    comparr = np.zeros((real.shape[0], 5))
    comparr[:, [0, 2]] = fit
    comparr[:, [1, 3]] = real
    comparr[:, 4] = w_fit
    print('   x_fit / x_real / y_fit / y_real / w_fit')
    print(round_digit(comparr, 4))