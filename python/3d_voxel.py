# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:43:20 2021

@author: jgreensl
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from utilities import utils,fio,constants

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()
    
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

# voxel plots only render edges, so need to explode
def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

# get coords from indices on exploded 
def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube0, elev=10, azim=320, min_value=3.0, figname=None, alphascale=0.15):
    """
    Assume 3darr is Z, Y, X [lev,lat,lon]
    INFO: time to run plot_cube  (19, 80, 192)  in seconds:  289.16
    """
    start = timer()

    ZDIM,YDIM,XDIM = cube0.shape
    # use normalized (0-1) cube for coloring
    normed = normalize(cube0)
    normed_rolled = np.transpose(normed,axes=[2,1,0]) # change z,y,x to x,y,z
    facecolors = cm.viridis(normed_rolled)
    # set alpha using value (max will be opaque)
    facecolors[:,:,:,-1] = normed_rolled * alphascale
    facecolors = explode(facecolors)
    # originally draw all nonzero
    #filled = facecolors[:,:,:,-1] != 0
    
    # we only draw where wind is > some min value
    # use orig for where we plot the voxels
    rolled = np.transpose(cube0,axes=[2,1,0]) # change z,y,x to x,y,z
    rolled = explode(rolled)
    filled = rolled > min_value
    x,y,z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca(projection='3d')
    ax.view_init(elev, azim)
    ax.set_xlim(right=XDIM*2)
    ax.set_ylim(top=YDIM*2)
    ax.set_zlim(top=ZDIM*2)
    
    ax.voxels(x, y, z, filled, facecolors=facecolors)#, shade=False)

    end = timer()
    print("INFO: time to run plot_cube ",cube0.shape," in seconds: ",end - start)
    if figname is None:
        plt.show()
    else:
        plt.title(figname)
        plt.savefig(figname)
        plt.close()
        print("INFO: ",figname," saved!")
    end2 = timer()
    print("INFO: time to save/display plot_cube ",cube0.shape," in seconds: ",end2 - end)

### TEST EXAMPLES:
def examples():
    filled = np.array([
        [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
        [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
    ])
    
    ax = make_ax(True)
    ax.voxels(filled, edgecolors='gray', )#shade=False)
    #plt.show()
    plt.savefig("3d_voxel_test1.png")
    plt.close()
    
    # add alpha channel to facecolors
    ax = make_ax(True)
    ax.voxels(filled, facecolors='#1f77b430', edgecolors='gray', )#shade=False)
    #plt.show()
    plt.savefig("3d_voxel_test2.png")
    plt.close()
    
    # colored internal cube, need to explode
    ax = make_ax()
    colors = np.array([[['#1f77b430']*3]*3]*3)
    colors[1,1,1] = '#ff0000ff'
    colors = explode(colors)
    filled = explode(np.ones((3, 3, 3)))
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=colors, edgecolors='gray')#, shade=False)
    #plt.show()
    plt.savefig("3d_voxel_test3.png")
    plt.close()


# read 3d winds, destagger, plot cube
DS = fio.read_model_run_hour("KI_run1_exploratory",hour=1)
topog=DS['topog'] # [lat, lon]
wind_u0 = DS['wnd_ucmp'][0,::2,::2,::2] # time step 0
wind_v0 = DS['wnd_vcmp'][0,::2,::2,::2] # up to level 10
wind_u,wind_v = utils.destagger_winds(wind_u0.data,wind_v0.data)
windspeed=np.sqrt(wind_u**2+wind_v**2)


cloud = DS['cld_water'][0,::2,::2,::2] + DS['cld_ice'][0,::2,::2,::2]

plot_cube(cloud.data, min_value=.0001, figname="3d_voxel_cloud.png")

plot_cube(windspeed, min_value=10, figname="3d_voxel_windspeed.png")
plot_cube(wind_v, min_values=0.5, figname="3d_voxel_wind_v.png")
plot_cube(wind_u, figname="3d_voxel_wind_u.png")


