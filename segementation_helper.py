# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:42:44 2022

@author: Li
"""
#%%
import napari
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
#%%

#turn mrc file into a numpy 
#need to install the mcrfile with pip 
import mrcfile
with mrcfile.open('C:/Users/Li/Desktop/project/tomography_lchan/22_01_18_recon_slice256_512.mrc') as mrc: #using absolute file pathway 
    reconData = mrc.data
mrc.close()

#%%

viewer = napari.view_image(reconData)
napari.run()

#use paint brush tool and bucket. opacity change to 0.56

#%%
#conversion of tiff save to numpy array 
import dxchange
stack = dxchange.reader.read_tiff('12-27.tif')
import matplotlib.pyplot as plt 
plt.imshow(stack[41,:,:])
#%%

