# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Xeni Deligianni
Calculate Displacement Vectors from PC data
"""

#from pathlib import Path
import os
#from dicomUtils.dicom3D import load3dDicom
import numpy as np
from mrprot import read_siemens_prot
from scipy import ndimage, misc
from scipy.interpolate import RectBivariateSpline
import math 

def calc_disp(flow_x,flow_y,flow_z,info,mask):
    
    #Function for interpolating coordinates-WHY???????????
    def calcInterpolatedIndex(x,y,ActInterpFacX,ActInterpFacY):

        interpX = round( (x-1)*ActInterpFacX  + 1)
        interpY = round( (y-1)*ActInterpFacY  + 1)
        #Adapted for python-to check
        if (interpX < 0):
             interpX = 0
        elif interpX > (len(interpDispX)-1):
             interpX = len(interpDispX)-1
                
        #To check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (interpY < 0):
             interpY = 0
        elif  interpY>(len(interpDispY[0])-1):
             interpY = len(interpDispY[0])-1
                      
        #      print(interpX)
        return interpX,interpY 
            
    
    #Read info from siemens header
    prot = read_siemens_prot(info[0])
    nPhases = int(info[1]['CardiacNumberOfImages'].value)
    nPoints = np.count_nonzero(mask)
    SliceThickness= info[1]['SliceThickness'].value
    dt= float(info[1]['RepetitionTime'].value)#ms
    print(dt)
    TE = np.ones((1, nPhases))*dt# ms

    x = np.zeros((nPoints,nPhases))
    y = np.zeros((nPoints,nPhases))  
    u = np.zeros((nPoints,nPhases))
    v = np.zeros((nPoints,nPhases)) 
    w = np.zeros((nPoints,nPhases))
    dx = np.zeros((nPoints,nPhases))
    dy = np.zeros((nPoints,nPhases))
    dz = np.zeros((nPoints,nPhases))
    
    dimx=flow_x.shape[0]#height (len(flow_x))
    dimy=flow_x.shape[1]#width (len(flow_x[0]))
    dimz=len(flow_x[0][0])
    
    dispVx = np.zeros((dimx,dimy,dimz))
    dispVy = np.zeros((dimx,dimy,dimz))
    dispVz = np.zeros((dimx,dimy,dimz))
    
    ##CHECK WHY???? 10*0.5
    diffDispX = 10*flow_x*0.5*dt/info[1]['PixelSpacing'][0]/1000 # conversion cm->mm and ms->s
    diffDispY = 10*flow_y*0.5*dt/info[1]['PixelSpacing'][1]/1000
    diffDispZ = 10*flow_z*0.5*dt/SliceThickness/1000

    #Precalculate interpolated maps
    x = np.linspace(0,dimx,dimx)#height
    y = np.linspace(0,dimy,dimy)#width
   
    xx, yy = np.meshgrid(y, x)
    
    xnew = np.arange(1,dimx,0.1)
    ynew = np.arange(1,dimy,0.1)
   
    
    [Xq,Yq] = np.meshgrid(np.arange(1,dimy,0.1),np.arange(1,dimx,0.1))
        
    interpDispX = np.zeros((len(Xq),len(Xq[0]),nPhases))
    interpDispY = np.zeros((len(Xq),len(Xq[0]),nPhases))         
    interpDispZ = np.zeros((len(Xq),len(Xq[0]),nPhases))  
    
    for iph in range (nPhases):
        fX = RectBivariateSpline(x,y,diffDispX[:,:,iph])
        fY = RectBivariateSpline(x,y,diffDispY[:,:,iph])
        fZ = RectBivariateSpline(x,y,diffDispZ[:,:,iph])
        
        interpDispX[:,:,iph] = fX(xnew,ynew)
        interpDispY[:,:,iph] = fY(xnew,ynew)
        interpDispZ[:,:,iph] = fZ(xnew,ynew)    
    
    
    ActInterpFacX = len(interpDispX)/len(diffDispX)
    ActInterpFacY = len(interpDispY[0])/len(diffDispY[0])


    for iph in range (1,nPhases):
        for ix in range(flow_x.shape[0]):
            for iy in range(flow_x.shape[1]):
                if mask[ix,iy] == True :        
                          
                    newXn = ix + dispVx[ix,iy,iph]
                    newYn = iy + dispVy[ix,iy,iph]
        
                    newXnm1 = ix - dispVx[ix,iy,iph-1]
                    newYnm1 = iy - dispVy[ix,iy,iph-1]

                    #if((math.isnan(newXnm1) == False) & (math.isnan(newYnm1) == False)):
                        
                    inewXnm1, inewYnm1 = calcInterpolatedIndex(newXnm1, newYnm1,ActInterpFacX,ActInterpFacY)
                    inewXn, inewYn = calcInterpolatedIndex(newXn, newYn,ActInterpFacX,ActInterpFacY)
                        
                    deltaDispX = interpDispX[inewXnm1,inewYnm1,iph-1]+ interpDispX[inewXn,inewYn,iph]
                    deltaDispY = interpDispY[inewXnm1,inewYnm1,iph-1]+ interpDispY[inewXn,inewYn,iph]
                    deltaDispZ = interpDispZ[inewXnm1,inewYnm1,iph-1]+ interpDispZ[inewXn,inewYn,iph]
                        
                    dispVx[ix,iy,iph] = dispVx[ix,iy,iph-1] + deltaDispX
                    dispVy[ix,iy,iph] = dispVy[ix,iy,iph-1] + deltaDispY
                    dispVz[ix,iy,iph] = dispVz[ix,iy,iph-1] + deltaDispZ
                   

# output displacement is in mm
    
    print("DispVx shape", dispVx.shape)       

    dispVxi = info[1]['PixelSpacing'].value[0] * dispVx
    
    print("DispVxi shape", dispVxi.shape)       
    
    dispVyi = np.multiply(info[1]['PixelSpacing'].value[1],dispVy)
    dispVzi = np.multiply(info[1]['SliceThickness'].value,dispVz)
    
    return dispVxi, dispVyi, dispVzi

