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
%matplotlib qt
from layerizationHelpers import fitPlaneLTSQ, measureDistFromPlane, layerClustering

#%%
#making COM Layers - now outside of function 
df = pd.read_csv('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv', header=None) 
dfnp = pd.DataFrame.to_numpy(df) 
comLayer = np.zeros((np.shape(dfnp)[0],12))
comLayer[:,1] = dfnp[:,0]
comLayer[:,2] = dfnp[:,1]
comLayer[:,3] = dfnp[:,2]
#%%
store100 = measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_100.csv') #needs to be reselected in MAT
#%%
defaultPixelSize = 0.095*4

comLayer[:,4] = layerClustering(defaultPixelSize,8,6.5,-480,store100)

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
store010 = measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_010.csv')
#%%
comLayer[:,5] = layerClustering(defaultPixelSize,8,6.5,-390, store010)

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
store001 = measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_001.csv', z_buffer =-500) #needs to be reselected in MAT
#%%
comLayer[:,6] = layerClustering(defaultPixelSize, 14, 6.5, -515, store001)

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
store110 = measureDistFromPlane('C:/Users/Li/Desktop/git/tomographyscripts_lchan/particle_final_copy.csv','cursor_temp_110.csv') #needs to be reselected in MAT
#%%
comLayer[:,7] = layerClustering(defaultPixelSize, 17, 6.5/np.sqrt(2), -1750, store110)

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