#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:19:09 2018

@author: bombus
"""

#import os
import numpy as np
#from PIL import Image
#import scipy.misc
#import scipy.ndimage






def vector_length(vector):
    """
    Input: a vector
    Output: the euclidean length of the vector
    """
    return np.sqrt(np.sum(np.square(vector.astype('float64'))))

#def same_scale_new_2feb(image,ref_image):
#    """
#    Input: two np.arrays that represent RGB images.
#    output: an input of the size of the image such that the vector lengths
#    are in the same range as the reference image
#    """
#    #the shapes
#    image_shp=image.shape
#    ref_shp=ref_image.shape
#    #and array to put the length in, as basic number 220 is chosen to avoid
#    #division through 0
#    image_len=220*np.ones(image_shp[:2])
#    ref_len=np.zeros(ref_shp[:2])
#    for row in range(image_shp[0]):
#        for col in range(image_shp[1]):
#            if image[row,col,:].tolist() != [0,0,0]:
#                image_len[row,col]=vector_length(image[row,col,:])              
#    for row in range(ref_shp[0]):
#        for col in range(ref_shp[1]):
#            ref_len[row,col]=vector_length(ref_image[row,col,:])
#    #correct the shape of the image_len array
#    image_len=np.array([image_len,image_len,image_len]).transpose(1,2,0)
#    #pick the parametes
#    n1,m1=np.min(image_len),np.min(ref_len)        
#    n2,m2=np.max(image_len),np.max(ref_len)
#    #tha scale and the translation factors.
#    factor = (m1-m2)/(n1-n2)
#    basis = (m2*n1-m1*n2)/(n1-n2)
#    #the new image in the correct shape
#    new_image=factor*image+basis*image/image_len
#    new_image=new_image.astype('uint8')
#    return new_image#,factor, basis

def same_scale_new(image,ref_image):
    """
    Input: two np.arrays that represent RGB images.
    output: an input of the size of the image such that the vector lengths
    ar in the same range as the reference image
    """
    #the shapes
    image_shp=image.shape
    ref_shp=ref_image.shape
    #and array to put the length in, as basic number 220 is chosen to avoid
    #division through 0
    image_len=220*np.ones(image_shp[:2])
    ref_len=np.zeros(ref_shp[:2])
    for row in range(image_shp[0]):
        for col in range(image_shp[1]):
            if image[row,col,:].tolist() != [0,0,0]:
                image_len[row,col]=vector_length(image[row,col,:])              
    for row in range(ref_shp[0]):
        for col in range(ref_shp[1]):
            ref_len[row,col]=vector_length(ref_image[row,col,:])
    #correct the shape of the image_len array
    image_len=np.array([image_len,image_len,image_len]).transpose(1,2,0)
    #pick the parametes
    n1,m1=np.min(image_len),np.min(ref_len)        
    n2,m2=np.max(image_len),np.max(ref_len)
    #the scale and the translation factors.
    factor = (m1-m2)/(n1-n2)
    basis = (m2*n1-m1*n2)/(n1-n2)
    #the new image in the correct shape
    if factor<1.01 and basis<50:
        new_image=factor*image+basis*image/image_len
        new_image=new_image.astype('uint8')
    else:
        new_image=image
        new_image=new_image.astype('uint8')
    return new_image#,factor, basis


#def glue_new_old(grd_img,bee_img,ymin,xmin):
#    """
#    inputs: grd_image an array representing the image in which the array called
#    bee_img has to bee glued at a location detremined by ymin,xmin
#    output: and array representing the new iamge. 
#    """
#    #the shapes of the ground images and the image of the bees
#    bee_shp=bee_img.shape
#    grd_shp=grd_img.shape
#    #the part of the ground image on which the bee will be glued.
#    img_around_glue = grd_img[ymin-10:ymin+bee_shp[0]+10,xmin-10:xmin+bee_shp[0]+10,:]
#    new_bee_img=same_scale_new(bee_img,img_around_glue)
#    #bee_img_sc=scale_image(bee_img,factor)
#    for row in range(grd_shp[0])[ymin:ymin+bee_shp[0]]:
#        for col in range(grd_shp[1])[xmin:xmin+bee_shp[1]]:
#            by,bx=row-ymin,col-xmin
#            #alpha is almost one in the middle of the bee_img and zero on the boundary
#            alpha=by*(bee_shp[0]-by)*bx*(bee_shp[1]-bx)*16/(bee_shp[0]**2*bee_shp[1]**2)
#            if bee_img[by,bx,:].tolist() != [0,0,0]:
#                grd_img[row,col,:] = alpha*new_bee_img[by,bx,:]+(1-alpha)*grd_img[row,col,:]
#    grd_img=grd_img.astype('uint8')            
#    ymin,ymax,xmin,xmax=ymin,min(grd_shp[0],ymin+bee_shp[0]),xmin,min(grd_shp[1],xmin+bee_shp[1])            
#    return grd_img,(ymin,ymax,xmin,xmax)   

#def weights_old(twoDshape):
#    shape= twoDshape
#    w=np.ones(shape)
#    for row in list(range(shape[0])[:6]) + list(range(shape[0])[-6:]):
#        for col in range(shape[1]):
#            w[row,col]=row*(shape[0]-row)*col*(shape[1]-col)*16/(shape[0]**2*shape[1]**2)
#    for col in list(range(shape[1])[:6])+list(range(shape[1])[-6:]):
#        for row in range(shape[0]):
#            w[row,col]=row*(shape[0]-row)*col*(shape[1]-col)*16/(shape[0]**2*shape[1]**2)
#    return w        

def weights(twoDshape):
    shape= twoDshape
    w=np.ones(shape)
    for row in list(range(shape[0])[:6]):
        for col in range(shape[1]):
            w[row,col]=min(1,row/6.0)
    for col in list(range(shape[1])[:6]):
        for row in list(range(shape[0])):
            w[row,col]=min(w[row,col],col/6.0)
    for row in list(range(shape[0])[-6:]):
        for col in range(shape[1]):
            w[row,col]=min(w[row,col],(shape[0]-row)/6.0)
    for col in list(range(shape[1])[-6:]):
        for row in range(shape[0]):
            w[row,col]=min(w[row,col],(shape[1]-col)/6.0)        
    return w        

            
def glue_new(grd_img,bee_img,ymin,xmin):
    """
    inputs: grd_image an array representing the image in which the array called
    bee_img has to bee glued at a location detremined by ymin,xmin
    output: and array representing the new iamge. 
    """
    #the shapes of the ground images and the image of the bees
    bee_shp=bee_img.shape
    grd_shp=grd_img.shape
    #the part of the ground image on which the bee will be glued.
    img_around_glue = grd_img[ymin-10:ymin+bee_shp[0]+10,xmin-10:xmin+bee_shp[0]+10,:]
    new_bee_img=same_scale_new(bee_img,img_around_glue)
    # the weigths for weighted addeition
    w = weights(bee_shp[:2])
    #print(new_bee_img.shape)
    #bee_img_sc=scale_image(bee_img,factor)
    for row in range(grd_shp[0])[ymin:ymin+bee_shp[0]]:
        for col in range(grd_shp[1])[xmin:xmin+bee_shp[1]]:
            by,bx=row-ymin,col-xmin
            #alpha is almost one in the middle of the bee_img and zero on the boundary
            #alpha=by*(bee_shp[0]-by)*bx*(bee_shp[1]-bx)*16/(bee_shp[0]**2*bee_shp[1]**2)
            if bee_img[by,bx,:].tolist() != [0,0,0]:
                #grd_img[row,col,:] = new_bee_img[by,bx,:]+grd_img[row,col,:]
                grd_img[row,col,:] = w[by,bx]*new_bee_img[by,bx,:]+(1-w[by,bx])*grd_img[row,col,:]
    grd_img=grd_img.astype('uint8')            
    ymin,ymax,xmin,xmax=ymin,min(grd_shp[0],ymin+bee_shp[0]),xmin,min(grd_shp[1],xmin+bee_shp[1])            
    return grd_img,(ymin,ymax,xmin,xmax)

#%%
"""
Test the results
"""    
import os
from PIL import Image 
import matplotlib.pyplot as plt   
PATH_bees='/home/bombus/Dropbox/HommelsObservaties/Synthetic_image_generation/cut_bees_webcam/'
PATH_flowers='/home/bombus/flowers/images/'

bee_files = os.listdir(PATH_bees)
bee_npy = [a for a in bee_files if a.endswith('npy')]
bee_images = [np.load(PATH_bees+a) for a in bee_npy]

flower_files=os.listdir(PATH_flowers)
flower=Image.open(PATH_flowers+flower_files[1028])
flower=np.array(flower)

#
#
#
#
##%%
#for im in bee_images:
#    ssn=same_scale_new(im,dahlia)
#    plt.figure()
#    plt.imshow(ssn)
#    plt.show()
##%%
ni=glue_new(flower,bee_images[2],150,100) 
plt.imshow(ni[0])        
##
###%%
###save ssn as png and load it again
##scipy.misc.imsave('test.png',ssn)
###%%
### de boot is met een ander programma gemaakt daarom kan dat zo.
##boot=scipy.ndimage.imread('Boot1.png')
##
###%%









