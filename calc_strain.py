#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:58:28 2021
Calculate 2D strain and strain rate
@author: xenia
source for function in Matlab:
    Khaled Z et al. "Direct Three-Dimensional Myocardial Strain Tensor 
   Quantification and Tracking using zHARP"
initial Function in matlab is written by D.Kroon University of Twente (February 2009)
"""
import os
from dicomUtils.dicom3D import load3dDicom
import numpy as np
import math 
# https://github.com/espdev/sgolay2
from mpl_toolkits.mplot3d import Axes3D
from sgolay2d import sgolay2d
from numpy import linalg as LA

def calc_strain(dispX,dispY,dispZ,mask,fx,fy):
    
    def strain2D(dispX_2D, dispY_2D):
        
        dimx=fx.shape[0]#height (len(flow_x))
        dimy=fx.shape[1]#width (len(flow_x[0]))
        dimz=len(fx[0][0])
        
        E= np.zeros((dimx,dimy,2,2))
               
        Uxx = sgolay2d(dispX_2D, window_size=13, order=4, derivative="col")
        Uxy = sgolay2d(dispX_2D, window_size=13, order=4, derivative="row")
    
        Uyx = sgolay2d(dispY_2D, window_size=13, order=4, derivative="col")
        Uyy = sgolay2d(dispY_2D, window_size=13, order=4, derivative="row")
        
        for ix in range(dispX_2D.shape[0]):
            for iy in range(dispX_2D.shape[1]):
                
                # The displacement gradient
                Ugrad = np.array([[Uxx[ix,iy], Uxy[ix,iy]], [Uyx[ix,iy], Uyy[ix,iy]]])
                
                # The (inverse) deformation gradient
                Finv = np.array([[1,0],[0,1]])-Ugrad #%F=inv(Finv); 
                
                # The 3-D Eulerian strain tensor
                e=(1/2)*(np.array([[1,0],[0,1]])-Finv*Finv.transpose())
                # Store tensor in the output matrix
                E[ix,iy,:,:]=e
                
        return E
        

    dimx=fx.shape[0] #height (len(flow_x))
    dimy=fx.shape[1] #width (len(flow_x[0]))
    dimz=len(fx[0][0])
    
    Eig_v = np.zeros((dimx,dimy,2,dimz))
    s = np.zeros((dimx,dimy)) 

    for iz in range(1,dimz):
        
        s= strain2D(dispX[:,:,iz],dispY[:,:,iz])  
        
        for ix in range(dispX.shape[0]):
            for iy in range(dispX.shape[1]):
                
                if mask[ix,iy] == True :
                    
                    #2D
                    # e1 (stretching), e2 (compression)
                    sorted_array = np.sort(LA.eigvals(np.squeeze(s[ix,iy,:,:])))
                    Eig_v[ix,iy,:,iz]  = sorted_array[::-1]
                    
        
    return Eig_v