#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:56:35 2018
A class that is depended on a Keras model. With it you can sligth over an image 
and it scans the image for bumblebees present, draws a (bounding)box  where it finds one. 

@author: bombus
"""

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy import ndimage as ndi

#model_classifyer4_april werkt eigenlijk het beste
# load json and create model
json_file = open('classifyer/model_classifyer4_april.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("classifyer/model_classifyer4_april.h5")
print("Loaded model from disk")

def metric(array1,array2):
    yd=abs(array1[0]-array2[0])
    xd=abs(array1[1]-array2[1])
    return max(yd,xd)
    
class ImageWithBumblebees():
    def __init__(self,filename):
        self.filename = filename
    def array(self): 
        return img_to_array(load_img(self.filename))/255
    def array_exp(self):
        return np.expand_dims(img_to_array(load_img(self.filename))/255, axis=0)
    def scan_preds(self,n):
        img = self.array_exp()
        shp = img.shape
        predictions=np.zeros(((shp[1]-45)//n,(shp[2]-45)//n))  
        
        for row in range(predictions.shape[0]):
            for col in range(predictions.shape[1]):
                pred = model.predict(img[:,n*row:n*row+45,n*col:n*col+45,:])
                predictions[row,col] = pred
        return predictions

    def scan(self,n,tresshold):
        img = self.array_exp()
        shp = img.shape
        predictions=np.zeros(((shp[1]-45)//n,(shp[2]-45)//n))  
        
        for row in range(predictions.shape[0]):
            for col in range(predictions.shape[1]):
                pred = model.predict(img[:,n*row:n*row+45,n*col:n*col+45,:])
                if pred > tresshold:
                    predictions[row,col] = pred
        return predictions
    def location_labels(self,n,tresshold):
        preds= self.scan(n,tresshold)
        mask=np.zeros((preds.shape))
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if preds[row,col] > 0:
                    mask[row,col] =1
        all_labels,number_labels = ndi.label(mask)
        return all_labels,number_labels,preds
    def locations(self,n,tresshold):
        locs,nlocs,preds = self.location_labels(n,tresshold)
        locations = []
        for location in range(nlocs+1)[1:]:
            npa=np.argwhere(locs==location)
            #print(npa.shape)
            l = npa[npa.shape[0]//2,:]
            locations.append(l)
        true_locations = [n*a for a in locations]
        return locations, true_locations, preds
    def locations_cleaned(self,n,tresshold,mindist):
        l,tl,preds = self.locations(n,tresshold)
        l=[list(a) for a in l]
        m= l[:1]
        for loc in l[1:]:
            distances=[metric(loc,a) for a in m]
            if min(distances)>mindist//n:
                m.append(loc)
            elif min(distances)<mindist//n:
                loc1 = [a for a in m if metric(loc,a)<mindist//n]#list of too close points
                if len(loc1) > 1:
                    m = m
                elif len(loc1) ==1:
                    loc1=loc1[0]
                    #print(type(loc1))
                    if preds[loc[0],loc[1]]>preds[loc1[0],loc1[1]]:
                        #print(m,loc1,type(loc1))
                        m.append(loc)
                        m.remove(loc1)
                        #print('vervangen')
        true_m=[[n*b for b in a] for a in m]              
        return m,true_m,preds 
    def plot_found_loc(self,n,tresshold=0.3,mindist=40):
        locs,true_locs,preds=self.locations_cleaned(n,tresshold,mindist)
        img = self.array()*255
        img = img.astype('uint8')
        #print(type(img))
        fig,ax = plt.subplots(1)
    
        # Display the image
        ax.imshow(img)
        for loc in true_locs:    
            rect1 = patches.Rectangle((loc[1],loc[0]),45,45,
                             linewidth=1,edgecolor='r',facecolor='none')
    
            ax.add_patch(rect1)
        plt.show()
        #return img,preds
    def save_figure(self,exportname,n,tresshold=0.3,mindist=40):
        locs,true_locs,preds=self.locations_cleaned(n,tresshold,mindist)
        img = self.array()*255
        img = img.astype('uint8')
        #print(type(img))
        fig,ax = plt.subplots(1)
        # remove the axes
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # Display the image
        ax.imshow(img)
        for loc in true_locs:    
            rect1 = patches.Rectangle((loc[1],loc[0]),45,45,
                             linewidth=1,edgecolor='r',facecolor='none')
    
            ax.add_patch(rect1)
        fig.savefig(exportname,bbox_inches="tight")
        fig.clf()
        #return img
        
