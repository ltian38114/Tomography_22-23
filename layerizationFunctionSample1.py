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
    df = pd. read_csv(comLocation, header=None) 
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
    
#%%   
    #new function - layerization part 2 ends at 194 retruns new labels 
    # using the def to input
    #six input vars. 4 vars in 133, and the last 2 in first function 
    #out put of next function will be the labels - return lables 
    #labels from this will be = to layerzation part 2
    
def layerClustering(pixelSize, layerAmount, layerSpacing, layerOffset):
    """
    pixel size, float\n 
    layer amounts, integer \n,
    layer spacing, float \n 
    layer offset, integer \n, 
    
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
    kmeans = KMeans(n_clusters =inputClusterNumber).fit(store) #this complied fine

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

#%%
#making COM Layers - now outside of function 

comLayer = np.zeros((np.shape(dfnp)[0],12))
comLayer[:,1] = dfnp[:,0]
comLayer[:,2] = dfnp[:,1]
comLayer[:,3] = dfnp[:,2]
#%%
measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_100.csv') #needs to be reselected in MAT
#%%
defaultPixelSize = 0.095*4

comLayer[:,4] = layerClustering(defaultPixelSize,8,6.5,-480)

# Plotting and coloring the Layers

figX = plt.figure()
ax = figX.add_subplot(111,projection='3d')

colorMap = cm.tab20.colors

maxXLayer = int(np.amax(comLayer[:,4]))

for layerNum in range(1,maxXLayer+1):
    if 1 == 1:
       thisLayer = comLayer[np.where(comLayer[:,4] == layerNum)]
       xVals = thisLayer[:,1]
       yVals = thisLayer[:,2]
       zVals = thisLayer[:,3]
       size = 200
    
       ax.scatter(xVals, yVals,zVals, s = size, color = colorMap[layerNum], label = layerNum)
ax.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0)


plt.title("[100]-SL Layering")    

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

#did not add in lines 252 and onwards (the gaussian mixture model portion). 

# Saving Layerization Figure 
figureloc ="./figs/"
plt.savefig(figureloc+'layerization_100.png') #there is no figure loc here so im not sure what to put in to save figure since itds in layerizationPart2
plt.show()


#%%
measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_010.csv')
#%%
comLayer[:,5] = layerClustering(defaultPixelSize,8,6.5,-390)

figY = plt.figure()
ax = figY.add_subplot(111,projection='3d')

colorMap = cm.tab20.colors

maxYLayer = int(np.amax(comLayer[:,5]))

for layerNum in range(1,maxYLayer+1):
    if 1 == 1:
       thisLayer = comLayer[np.where(comLayer[:,5] == layerNum)]
       xVals = thisLayer[:,1]
       yVals = thisLayer[:,2]
       zVals = thisLayer[:,3]
       size = 200
    
       ax.scatter(xVals, yVals,zVals, s = size, color = colorMap[layerNum], label = layerNum)
ax.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0)


plt.title("[010]-SL Layering")    

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

#did not add in lines 252 and onwards (the gaussian mixture model portion). 

# Saving Layerization Figure 
figureloc ="./figs/"
plt.savefig(figureloc+'layerization_010.png') #there is no figure loc here so im not sure what to put in to save figure since itds in layerizationPart2
plt.show()
#%%
measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_001.csv', z_buffer =-500) #needs to be reselected in MAT
#%%
comLayer[:,6] = layerClustering(defaultPixelSize, 14, 6.5, -515)

# Plotting and coloring the Layers

figZ = plt.figure()
ax = figZ.add_subplot(111,projection='3d')

colorMap = cm.tab20.colors
#tried viridis and inferno - got soloid color cubes. 
#tried changing the tab value 
#colorMap = cm.coolwarm - subscriplable error
#colorMap = plt.cm.get_cmap('viridis',12) - getting "listedcolormap' not subscriplable errror

maxZLayer = int(np.amax(comLayer[:,6]))

for layerNum in range(1,maxZLayer+1):
    if 1 == 1:
       thisLayer = comLayer[np.where(comLayer[:,6] == layerNum)]
       xVals = thisLayer[:,1]
       yVals = thisLayer[:,2]
       zVals = thisLayer[:,3]
       size = 200
    
       ax.scatter(xVals, yVals,zVals, s = size, color = colorMap[layerNum], label = layerNum)
ax.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0)


plt.title("[001]-SL Layering")    

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

#did not add in lines 252 and onwards (the gaussian mixture model portion). 

# Saving Layerization Figure 
figureloc ="./figs/"
plt.savefig(figureloc+'layerization_001.png') #there is no figure loc here so im not sure what to put in to save figure since itds in layerizationPart2
plt.show()
#%%
measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_110.csv') #needs to be reselected in MAT
#%%
comLayer[:,7] = layerClustering(defaultPixelSize, 17, 6.5/np.sqrt(2), -1750)

# Plotting and coloring the Layers

figXY = plt.figure()
ax = figXY.add_subplot(111,projection='3d')

colorMap = cm.tab20.colors

maxXYLayer = int(np.amax(comLayer[:,7]))

for layerNum in range(1,maxXYLayer+1):
    if 1 == 1:
       thisLayer = comLayer[np.where(comLayer[:,7] == layerNum)]
       xVals = thisLayer[:,1]
       yVals = thisLayer[:,2]
       zVals = thisLayer[:,3]
       size = 200
    
       ax.scatter(xVals, yVals,zVals, s = size, color = colorMap[layerNum], label = layerNum)
ax.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0)


plt.title("[110]-SL Layering")    

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

#did not add in lines 252 and onwards (the gaussian mixture model portion). 

# Saving Layerization Figure 
figureloc ="./figs/"
plt.savefig(figureloc+'layerization_110.png') #there is no figure loc here so im not sure what to put in to save figure since itds in layerizationPart2
plt.show()



    #everything after will not be a function and be own script 
#layerizationPart1('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_110.csv') 
#'cursor_temp_110.csv' 
#r'C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv'