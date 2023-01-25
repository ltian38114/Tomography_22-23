# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:36:44 2022

@author: Li
"""

#importing packages 
import numpy as np 
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.io as pio 
pio.renderers.default = 'browser'
import os 
from sklearn.cluster import KMeans

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


def measureDistFromPlane(comLocation, selectedLocation, z_buffer = -1500, saveFigure= True, figureloc = "./figs/"):
    """
    COM data file location, string\n
    selected layer file location, string\n
    Z Buffer, integer 
    
    """
    #looks and makes folder if not already existing - all figures will now go here
    if not (os.path.isdir(figureloc)):
        os.mkdir(figureloc)

    
    #importing COM data 
    df = pd.read_csv(comLocation, header=None) 
    dfnp = pd.DataFrame.to_numpy(df) 
    #importing selected data
    cursor_data = pd.read_csv(selectedLocation, header = None )
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
    figDefaultName = selectedLocation.split(".")[-2] #splits selected location into list by all periods in name and selects -2 width 
    
    offSetFigName = figDefaultName+"offset_figure.png"
    plt.savefig(figureloc+offSetFigName)
    plt.show()
    
    #need to do distance calculations now then used for histagram 

    xVals = dfnp[:,0] 
    yVals = dfnp[:,1]
    zVals = dfnp[:,2]

    store = np.zeros((np.shape(dfnp)[0], 4))

    p1 = np.array([xx[0,0], yy[0,0], z[0,0]]) # equivalent to the plane 


    for range_num in range(np.shape(dfnp)[0]):
        p0 = np.array([dfnp[range_num,0],dfnp[range_num,1], dfnp[range_num,2]])
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
    plt.savefig(figureloc+figDefaultName+'histogram.png') 
    plt.show()
    
    return store
    
#%%   
    #new function - layerization part 2 ends at 194 retruns new labels 
    # using the def to input
    #six input vars. 4 vars in 133, and the last 2 in first function 
    #out put of next function will be the labels - return lables 
    #labels from this will be = to layerzation part 2
    
def layerClustering(pixelSize, layerAmount, layerSpacing, layerOffset, store):
    """
    pixel size, float\n 
    layer amounts, integer \n,
    layer spacing, float \n 
    layer offset, integer \n, 
    store
    """
    #pixelSize = 0.095*4 
    #layerAmount = 17
    #layerSpacing = 6.5 or 6.5/rt(2) nm per layer
    
    #calculate
    layerThickness = layerSpacing*layerAmount

    pixelLayerThickness = np.ceil(layerThickness/pixelSize) #np.ceil round up 

    guessInitialZ = np.linspace(layerOffset,(layerOffset+pixelLayerThickness), num=layerAmount)
    
    initialZ = guessInitialZ

    inputClusterNumber = np.shape(initialZ)[0]
    #inputClusterNumber = 16
    kmeans = KMeans(n_clusters =inputClusterNumber).fit(store)
    
    proj_list = store[:,3]
    proj_list_df = pd.DataFrame(proj_list)

    projlist_2D = np.zeros((np.shape(proj_list)[0],2))
    projlist_2D[:,0] = proj_list
    projlist_2D[:,1] = proj_list

    labels = kmeans.fit(projlist_2D).predict(projlist_2D) #creates vector of assignments --> attach to initial COM array to plot
    labels = labels + 1

    statArray = np.zeros((np.shape(np.unique(labels))[0],4))

    for thisParticle in range(0,np.shape(store)[0]):
        thisDistance = store[thisParticle, 3]
        thisLayer = labels[thisParticle] - 1
        statArray[thisLayer, 3] += thisDistance #first index is thisLayer, second index is the 0 and 1
        statArray[thisLayer, 1] += 1 #making array where first index has all the labels #second index has info 
        statArray[thisLayer, 2] = thisLayer + 1 #keeps track of indices

    #finding averages
        statArray[thisLayer,0] = statArray[thisLayer,3] / statArray[thisLayer, 1]

    agg = statArray[statArray[:,0].argsort()]


    numList = range(1,np.max(labels)+1)

    labelDict = dict(zip(agg[:,2],numList))

    newLabels = np.zeros(np.shape(labels))
    
    for eachLabel in range(0,np.shape(labels)[0]):
        newLabels[eachLabel] = labelDict[labels[eachLabel]]
    
    return newLabels
