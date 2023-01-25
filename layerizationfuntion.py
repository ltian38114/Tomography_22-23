# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:31:23 2022

"""
#importing packages 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd 
import plotly
import plotly.express as px
import plotly.io as pio 
pio.renderers.default = 'browser'
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os 

def fitPlaneLTSQ(data):
    (rows, cols) = data.shape
    G = np.ones((rows, 3))
    G[:, 0] = data[:, 0]  #X
    G[:, 1] = data[:, 1]  #Y
    Z = data[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)


def layerizationPart1(comLocation, selectedLocation, z_buffer = -1500, saveFigure= True, figureloc = "./figs/"):
    """
    COM data file location, string
    selected layer file location, string
    Z Buffer, integer 
    
    """
    #looks and makes folder if not already existing - all figures will now go here
    if not (os.path.isdir(figureloc)):
        os.mkdir(figureloc)

    
    #importing COM data 
    df = pd. read_csv(comLocation, header=None) 
    dfnp = pd.DataFrame.to_numpy(df) 
    #importing selected data
    cursor_data = pd.read_csv('cursor_temp_110.csv', header = None )
    cursor_data = np.array(cursor_data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colorMap = cm.tab20.colors

    data = cursor_data
    
    c, normal = fitPlaneLTSQ(cursor_data)

    # plot fitted plane
    maxx = np.max(data[:,0])
    maxy = np.max(data[:,1])
    minx = np.min(data[:,0])
    miny = np.min(data[:,1])

    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)

    # plot orginal data 607 --> all scatter data 
    ax.scatter(dfnp[:, 0], dfnp[:, 1], dfnp[:, 2])

    # compute needed points for plane plotting
    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    
    z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2] + z_buffer #tried adding in buffer here. 

    #plotting plane
    ax.plot_surface(xx, yy, z, alpha=0.2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    offSetFigName = "offset_figure.png"
    plt.savefig(figureloc+offSetFigName)
    plt.show()
    
    #need to do distance calculations now then used for histagram 

    xVals = dfnp[:,0] 
    yVals = dfnp[:,1]
    zVals = dfnp[:,2]

    store = np.zeros((np.shape(dfnp)[0], 4))

    p1 = np.array([xx[0,0], yy[0,0], z[0,0]]) # equivalent to the plane 


    for range_num in range(np.shape(dfnp)[0]):
        p0 = np.array([dfnp[range_num,0],dfnp[range_num,1], dfnp[range_num,2]])#thinking about making all of these equibvalent to the scatter
        dist = p0-p1
        proj = np.dot(normal, dist)
        store[range_num,0] = dist[0]
        store[range_num,1] = dist[1]
        store[range_num,2] = dist[2]
        store[range_num,3] = proj
        
       
    proj_list = store[:,3]
    proj_list_df = pd.DataFrame(proj_list)
    histFig = plt.figure()
    plt.hist(proj_list, bins = 50)
    
    #saving histogram
    plt.savefig(figureloc+'histogram.png') #saving the projection graph correctly not the histogram
    plt.show()
    
    
    
    
    
    
#r'C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv'