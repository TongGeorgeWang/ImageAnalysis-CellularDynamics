import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def ricker2d(x, y, a):
    return (1/a)*(2-((x/a)**2+(y/a)**2))*np.exp(-(x**2+y**2)/(2*a**2))

def make_kernel(w_kern, f, *func_args, show=True):
    ''' 
    - w_kern: width of the kernel in pixels. kernel centered at (0,0)
    - f, *func_args: the 2d function f(X, Y, *func_args)
    '''
    # To ensure kernel is centered at 0 in both even/odd cases    
    if w_kern % 2: # odd
        offset = 1
    else: # even
        offset = 0.5

    n_arr = np.arange(-w_kern//2, w_kern//2) + offset # ensures centered at 0

    X,Y = np.meshgrid(n_arr, n_arr)
    Z = f(X, Y, *func_args)

    if show:
        fig = plt.figure()
        fig.set(size_inches=fig.get_size_inches()*0.75)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_title(f'PSF Convolution kernel')
    
    return Z


def plot_kernel(w_kern, f, *func_args):
    ''' 
    - w_kern: width of the kernel in pixels. kernel centered at (0,0)
    - f, *func_args: the 2d function f(X, Y, *func_args)
    '''
    # To ensure kernel is centered at 0 in both even/odd cases    
    if w_kern % 2: # odd
        offset = 1
    else: # even
        offset = 0.5

    n_arr = np.arange(-w_kern//2, w_kern//2) + offset # ensures centered at 0

    X,Y = np.meshgrid(n_arr, n_arr)
    Z = f(X, Y, *func_args)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    
    return fig, ax


def wt_a(im, w_kern, a, **kwargs):
    Z = make_kernel(w_kern, ricker2d, a, show=False)
    WT_arr = sig.convolve2d(im, Z, mode='same', **kwargs) #, boundary='wrap')
    return WT_arr


def scalogram(im, n_kern, a_arr, crop_offset, scaling = 3):

    Es = []
    wt_s = []
    n = im.shape[0]

    for a in a_arr:
        print(a, flush=True)
        wt = wt_a(im, n_kern, a)
        wt = wt[crop_offset:n-crop_offset, crop_offset:n-crop_offset]
        Es.append(np.sum(wt**2))
        wt_s.append(wt)
    
    E_arr = (1/a_arr**scaling)*np.array(Es)
    wt_arr = np.array(wt_s)

    # plot
    fig, ax = plt.subplots()
    ax.plot(a_arr, E_arr)
    ax.set(title='Scalogram', xlabel='Wavelet scale $a$', ylabel = 'Energy')

    return E_arr, wt_arr


def scalogram_2(im, w_kern_arr, a_arr, crop_offset, scaling = 3):
    ''' 
    Difference is you can input w_kern as array :)
    '''

    Es = []
    wt_s = []
    n = im.shape[0]

    # if w_kern_arr.size != a_arr.size:
    #     raise IndexError('w_kern_arr wrong size')
    
    for i in range(a_arr.size):
        a = a_arr[i]
        w_kern = w_kern_arr[i]
        print(a, flush=True)
        wt = wt_a(im, w_kern, a)
        wt = wt[crop_offset:n-crop_offset, crop_offset:n-crop_offset]
        Es.append(np.sum(wt**2))
        wt_s.append(wt)
    
    E_arr = (1/a_arr**scaling)*np.array(Es)
    wt_arr = np.array(wt_s)

    # plot
    fig, ax = plt.subplots()
    ax.plot(a_arr, E_arr)
    ax.set(title='Scalogram', xlabel='Wavelet scale $a$', ylabel = 'Energy')

    return E_arr, wt_arr


def scalogram_max(im, a_arr, crop_offset, w_kern_arr = 10, scaling = 0):
    ''' 
    - Difference is you can input w_kern as array :)
    - auto set w_arr = 10*a_arr ...
    - return wt_arr, max_arr, E_arr
    '''

    Es = []
    wt_s = []
    n = im.shape[0]

    # if w_kern_arr.size != a_arr.size:
    #     raise IndexError('w_kern_arr wrong size')

    if type(w_kern_arr) == int:
        w_kern_arr = a_arr*w_kern_arr
    
    for i in range(a_arr.size):
        a = a_arr[i]
        w_kern = w_kern_arr[i]
        print(a, flush=True)
        wt = wt_a(im, w_kern, a)
        wt = wt[crop_offset:n-crop_offset, crop_offset:n-crop_offset]
        Es.append(np.sum(wt**2))
        wt_s.append(wt)
    
    E_arr = (1/a_arr**scaling)*np.array(Es)
    wt_arr = np.array(wt_s)
    max_arr = np.max(wt_arr, axis=(1,2))

    # plot
    fig, ax = plt.subplots()
    ax.plot(a_arr, max_arr)
    ax.set(title='Maximum wavelet coefficient at each scale', xlabel='Wavelet scale $a$', ylabel = 'Wavelet coefficient')

    return wt_arr, max_arr, E_arr