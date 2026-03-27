#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet filtering and SPT on eIF4E cells 
Jesse/Andrew's workflow, using Jesse/Andrew's code

Particle tracking analysis on eIF4E HILO data (imaging done on 2025-08-14). Fitting of the PSF is done using a Ricker wavelet (instead of a gaussian).

Step 2: After applying wavelet filter to image frames, run particle detection using TrackMate in ImageJ

Step 3 (this script): plot and analyze SPT trajectories 

File structure:

eIF4E-analysis/
├── code-eIF4E.ipynb
└── utils/
    └── helper.py
    └── plot.py
    └── spt.py
    └── wavelet.py
└── data/
    └── 32-2-crop-pos.csv
    └── Stream32-2.tif


"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import scipy.optimize as opt
import pandas as pd



from utils.helper import *


from utils import spt
from utils import plot

import tifffile as tiff


from sklearn.metrics import r2_score



"""
Import filtered image
"""

imgWL = tiff.imread('2_SPTdata/FilteredImage.tif')

# Imaging parameters
nFrames = 100
ExpInfo.scale = 0.094
ExpInfo.unit = '\mu m'
ExpInfo.dt = 10  # for 512x256 px images 
#ExpInfo.dt = 18.56  # for 512x512 px images 
ExpInfo.dt_unit = 'ms'




"""
Particle detection via thresholding
    the utility of wavelet filtering is that this is now feasible, where it is not in the 
    
    Dev: Thresholding might potentially be performed via elbow curves, and dense decomposition might be used to resolve >1 particle within a spot


pThresh = 4E6 

fig, ax = plot.show_imag(imgWL[0,:,:], cmap='gray')

pCMs = []
for n in range(0,nFrames):
    # Use cv2 to threshold the image, using the above threshold intensity 
    # Then regionprops to index thresholded particles 
    _, binary_image = cv2.threshold(imgWL[n,:,:], pThresh, 255, cv2.THRESH_BINARY)
    labeled_image = label(binary_image, connectivity=2, background=0)
    particles = regionprops(labeled_image)
    
    
    # Get centres of mass (CM) of indexed particles 
    particleCMs = np.zeros((len(particles),2))  
      

    for i, particle in enumerate(particles):
            #area = particle.area 
            particleCMs[i,:] = [particle.centroid[1], particle.centroid[0]] # (row, col) coordinates
    
            # Draw bounding box around particle in the master image 
            #bbox = particle.bbox # (min_row, min_col, max_row, max_col)
            #minr, minc, maxr, maxc = bbox
            #cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
    
    pCMs.append(particleCMs)
"""


"""
Define ROIs based on Centres of Mass (CM) of detected particles 
    ROI centre at particle CM
    ROI bounds (square) defined by roiWidth
    
roiWidth = 7


n4 = imgWL.shape[1] # img dimension
#cyc4 = CycleFunc(lambda i : plot.show_imag(imgWL[i]), imgWL.shape[0]) # cycler: each time it's run, will move to the next image 
#fig, ax = cyc4.move(0) 


for p in range(0,len(pCMs[0])):
#for p in range(0,10):
    start = pCMs[0][p,:]  
    
    ## Plot starting position, along with the selected ROI
    #ax.scatter(*start, marker='x', c='black') # mark particle of interest with x 
    plot.show_ROI(ax, *spt.get_ROI_3(*start, roiWidth, n4)) # draw box around ROI, 2nd input is ROI pixel width 

"""


        
"""
Import trajectories from TrackMate Excel file 
"""
df = pd.read_csv("2_SPTdata/Tracks.csv")
trajs = [g[['x','y']].to_numpy() 
                for _, g in df.groupby('ID')]

"""
Visualize trajectories 
"""
def plot_trajectories(trajs, background_image=None):
    """
    Plot all trajectories stored in 'trajs'.

    trajs: list of trajectories, where each trajectory is a list of (x,y) coordinates.
    background_image: optional 2D or 3D image to show underneath trajectory lines.
    figsize: figure size.
    """
    plt.figure()

    # Optional background image
    if background_image is not None:
        if background_image.ndim == 2:
            plt.imshow(background_image, cmap='gray')
        else:
            plt.imshow(background_image)
    
    # Plot each trajectory
    for traj in trajs:
        traj = np.array(traj)  # shape (T, 2)

        xs = traj[:,0]
        ys = traj[:,1]

        # Line
        plt.plot(xs, ys, linewidth=1)

        # Start & end markers
        #plt.scatter(xs[0], ys[0], marker='o', s=0.5)   # start
        #plt.scatter(xs[-1], ys[-1], marker='x', s=0.5) # end

    plt.title("Particle Trajectories")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    #plt.gca().invert_yaxis()  # because image coordinates have y downward
    plt.grid(False)
    plt.show()
    
plot_trajectories(trajs, background_image=imgWL[0])





"""
Trajectory viewing and Mean Squared Displacement computation 
"""

def MSD_curve_3_noStop(fp, show=True):
    ''' 
    - return t_arr (simply lag_arr in units of t), msd_curve
    ''' 
    #track_px = fp[:stop_i, 0:2]
    track_px = fp
    track_pos = spt.px2pos(track_px)
    N_steps = fp.shape[0]

    msd_curve = np.zeros(len(track_px)-1)
    lag_arr = np.arange(1, len(track_px))

    for m in lag_arr:
        msd_curve[m-1] = spt.MSD_lag_v2(track_pos, m)

    t_arr = lag_arr*ExpInfo.dt*10**-3

    if show:
        fig, ax = plt.subplots()
        ax.plot(t_arr, msd_curve, c = 'black', lw=1)
        ax.set(xlabel = f'Lag time [s]', ylabel = f'MSD [${ExpInfo.unit}^2$]')

    return t_arr, msd_curve



def add_track_noStop(ax, fp, **line_kwarg):
    track = fp
    line = Line2D(track.T[0], track.T[1], **default_kwargs(line_kwarg, c = 'white', lw = 1))
    ax.add_line(line)
    
    
    
def linear(x, a, b):
    return a*x + b

def linearZero(x, a, b):
    return a*x 

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

def MSD_curve_ij_noStop(fp, tj, i=2, j=5, show=True):

    track_px = fp
    track_pos = px2pos(track_px)

    lag_arr = np.arange(i, j+1)
    n_pts = j+1-i
    msd_curve = np.zeros(n_pts)

    for idx in range(n_pts):
        m = lag_arr[idx]
        msd_curve[idx] = MSD_lag(track_pos, m)

    t_arr = lag_arr*ExpInfo.dt*10**-3

    ## Fit

    popt, pcov = opt.curve_fit(linearZero, t_arr, msd_curve)
    R2 = r2_score(msd_curve, linearZero(t_arr, *popt)) #ydata,ypredicted; R2 calculation may not be correct 
    
    D = popt[0]/4 # for 2D diffusion 
    #D = popt[0]/6 # for 3D diffusion 

    if show==True:
        #fig, ax = plt.subplots()
        ax.scatter(t_arr, msd_curve,marker='.')
        ax.plot(t_arr, linearZero(t_arr, *popt), ls='--')
        #ax.set_xticks(t_arr)
        #ax.set_xticklabels(1000*t_arr)
        ax.set(xlabel = f'Lag time [{ExpInfo.dt_unit}]', ylabel = f'MSD [${ExpInfo.unit}^2$]')
        
        #if tj>=0:
            #ax.set_title(f'Traj{tj}: D ({i} to {j}) = {round(D, 4)} ${ExpInfo.unit}^2/s$')
        #else:
            #ax.set_title(f'D ({i} to {j}) = {round(D, 4)} ${ExpInfo.unit}^2/s$')
        #display(Math(f'D ({i} to {j}) = {round(D, 4)} {ExpInfo.unit}^2/s'))

    
    return D, R2#, t_arr, msd_curve, popt 



## View example trajectory 
trajToView = 2

track = trajs[trajToView] # track should be an Ntrack x 2 array of (x,y) coordinates, where Ntrack is the # of steps in the current trajectory 
track = np.array([track])[0] # convert track from list to array, in the right format 

t, msd = MSD_curve_3_noStop(track)
lims = spt.get_track_lims(track, offset=5)

fig, ax, _ = plot.show_cropped_im(imgWL[0], *lims) #, cmap='gray')
add_track_noStop(ax, track) #, c='black') #, lw=1)

avg_step = spt.track_dist(track).mean()

ax.set(
    title = f'Mean step size = {round(avg_step,2)} px ({round(avg_step*ExpInfo.scale,4)} ${ExpInfo.unit}$)',
    xlabel = 'x [px]', ylabel = 'y [px]'
)

MSD_curve_ij_noStop(track, tj=-1, i=1, j=len(track)-1, show=False) # i,j are the start and end points of the linear fit for D 




## Now loop through all trajectories and build statistics   
# Only take trajectories with greater than a threhsold number of points 
trajs_long = [elem for elem in trajs if len(elem) > 15]
trajs_badFits = [] # trajectories to omit 
trajs_good = [val for i, val in enumerate(trajs_long) if i not in trajs_badFits]


fig, ax = plt.subplots()
#plt.ylim(0,0.2)
#plt.xlim(0,0.1)

D = np.zeros([len(trajs_good),1])
R2 = np.zeros([len(trajs_good),1])
for tj in range(len(trajs_good)):
    track = trajs_good[tj] # track should be an Ntrack x 2 array of (x,y) coordinates, where Ntrack is the # of steps in the current trajectory 
    track = np.array([track])[0] # convert track from list to array, in the right format 

    t, msd = MSD_curve_3_noStop(track, show=False)
        
    #lims = spt.get_track_lims(track, offset=5)
    #fig, ax, _ = plot.show_cropped_im(imgWL[0], *lims) #, cmap='gray')
    #add_track_noStop(ax, track) #, c='black') #, lw=1)
    
    avg_step = spt.track_dist(track).mean()
    
    #ax.set(
    #    title = f'Mean step size = {round(avg_step,2)} px ({round(avg_step*ExpInfo.scale,4)} ${ExpInfo.unit}$)',
    #    xlabel = 'x [px]', ylabel = 'y [px]'
    #)
    
    D[tj],R2[tj] = MSD_curve_ij_noStop(track, tj, i=1, j=len(track)-1, show=True) # i,j are the start and end points of the linear fit for D 


## Plot D histogram 
fig, ax = plt.subplots()
plt.hist(D,bins=44)
ax.set(xlabel=f'D (${ExpInfo.unit}^2/s$)')

fig, ax = plt.subplots()
plt.hist(D,bins=22,range=(0,1))
ax.set(xlabel=f'D (${ExpInfo.unit}^2/s$)')



