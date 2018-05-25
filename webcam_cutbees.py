#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:42:37 2018
This file contains a sequence of functions with which you can find and cut out 
bees. 

As input you take a part of the larger image containing mostly one color, for example 
green, and as reference the same section of an image where there is no bee in there.


@author: bombus


"""

import numpy as np
import cv2
import copy
from skimage import  color
from scipy import ndimage as ndi





def difference(image,reference,tresshold=70):
    """
    takes a image and a reference image computes the difference and take those
    pixels from the image that differ a lot from the reference. The rest of the
    image is replaced by 1's.
    """
    diff=cv2.absdiff(image,reference)
    shp=diff.shape
    cut=np.ones(shp)
    for row in range(shp[0]):
        for col in range(shp[1]):
            sm=sum(abs(diff[row,col,:]))
            if sm > tresshold:
                cut[row,col,:]=image[row,col,:]            
    return cut.astype('uint8')            

def clean(image):
    """
    Goes over the rows of an image. If there is a ones vector in one of the neigboring
    pixels then it is set to 1's itself. 
    Subsequently it does the same while going over all columns.
    """
    shp=image.shape
    new_image=np.zeros(shp,dtype='uint8')
    
    
    #first clean the rows
    for row in range(shp[0]):
        for col in range(shp[1]):
            p=image[row,col-1:col+2,:].tolist()
            pf=[a for a in p if a ==[1,1,1]]
            if len(pf)<2:
                new_image[row,col] = image[row,col]
                
    for col in range(shp[1]):
        for row in range(shp[0]):
            p=image[row-1:row+2,col,:].tolist()
            pf=[a for a in p if a == [1,1,1]]
            if len(pf)<2:
                new_image[row,col] = image[row,col]
    return new_image            

def dominant_color(image):
    """
    gets the uint8 vector representing the dominant color in the image. 
    """
    arr = np.float32(image)
    pixels = arr.reshape((-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    return palette


def metric_colors2(col1,col2):
     """
     The distance between to colors using simple euclidean norm for the two 
     vectors after it is set lab. I don't know exactly what that means but it
     scales with human perception. 
     """
     col1_nom=color.rgb2lab(col1)
     col2_nom=color.rgb2lab(col2)
    
     col1_nom=col1_nom.flatten().flatten()
     col2_nom=col2_nom.flatten().flatten()
     diff=[]
     for ch in range(3):
         d=(col1_nom[ch]-col2_nom[ch])**2
         diff.append(d)
     pyth=sum(diff)
     pyth=np.sqrt(pyth)
     return pyth

def remove_dominant(image, cleaned_image,tresshold_color):
    """
    Two images and a tresshold. It determines the dominant color of the first
    image and subsequently removes sets pixels in cleaned image to one if the 
    distance is close to the dominant color, where close is detemined wrt to the
    tresshold.
    """
    diff=[]
    dominant = np.expand_dims(dominant_color(image),0)
    #print(dominant)
    new_image=copy.deepcopy(cleaned_image)
    #print(dominant)
    shp=cleaned_image.shape
    for row in range(shp[0]):
        for col in range(shp[1]):
            dept=np.expand_dims(cleaned_image[row,col,:],0)
            dept=np.expand_dims(dept,0)
            #print(dept)
            d=metric_colors2(dominant,dept)
            diff.append(d)
            if d<tresshold_color:
                new_image[row,col]=np.ones((3,))
    return new_image

def combine(image,reference,tresshold=50,tresshold_color=25):
    test=remove_dominant(reference,difference(image,reference,tresshold=tresshold),
                         tresshold_color=tresshold_color)
    test_clean=clean(test)
    return test_clean
                       
    

def mask(image):
    """
    creates an array of the same shape as the image that is 111 if the image is
    not 000, and the location where the bumble bee is found.
    """
    shp=image.shape
    msk=np.zeros(shp)
    for row in range(shp[0]):
        for col in range(shp[1]):
            
            if image[row,col].tolist() !=np.zeros((3,)).tolist():
                msk[row,col] =np.ones((3,))
    #segment the mask in connected components
    labels,_=ndi.label(msk)
    segments = [np.argwhere(labels==i) for i in range(_)]
    #get the component that is the one but largest
    segment_lengths=[a.shape[0] for a in segments]
    
    lengths= copy.deepcopy(segment_lengths)
    lengths.remove(max(lengths))
    #print(lengths)
    idx=segment_lengths.index(max(lengths))
    segment=segments[idx]
    #get the bounding box of that component
    ymin,ymax = np.min(segment[:,0]),np.max(segment[:,0])
    xmin,xmax = np.min(segment[:,1]),np.max(segment[:,1])
    #get an array that is 111 if a pixel is in the largest segment
    masker=np.zeros(shp)
    for a in segment.tolist():
        masker[a[0],a[1],a[2]]=1.0
    for i in range(2):    
        for row in range(shp[0]):
            for col in range(shp[1]):
                p=masker[row-5:row,col,:].tolist()
                q=masker[row:row+6,col,:].tolist()
                if [1,1,1] in p and [1,1,1] in q:
                    masker[row,col]=[1,1,1]
        for col in range(shp[1]):
            for row in range(shp[0]):
                p=masker[row,col-5:col,:].tolist()
                q=masker[row,col:col+5,:].tolist()
                if [1,1,1] in p and [1,1,1] in q:
                    masker[row,col]=[1,1,1]
    
    return masker,(ymin,ymax,xmin,xmax)            

def find_cut(image,reference,tresshold=50,tresshold_color=25):
    """
    Creates and array containing the bee embedded in an array of zeros.
    """
    res=combine(image,reference,tresshold=tresshold,tresshold_color=tresshold_color)
    msk,box=mask(res)
    ymin,ymax,xmin,xmax=box
    cut = (image*msk).astype('uint8')
    cut=cut[ymin:ymax,xmin:xmax,:]
    return cut

