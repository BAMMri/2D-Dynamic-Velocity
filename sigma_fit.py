#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:07:29 2021
Calculate build-up & Release Rates of strain
@author: Xeni Deligianni
"""
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import signal
import numpy as np
from numpy import nonzero as nz

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
from scipy.signal import savgol_filter

def calcRates(PosStrain,e1_Line,mask,info):
    
   def sigma_func(x,a,b,x0,dx):
        #a = params[0]
        #b = params[1]
        #x0 = params[2]
        #dx = params[3]
        y = b + (a-b)/(1+np.exp((x-x0)/dx))
        return y
   
   def showFit(timebase, yValues, modelFun, params):
      # plt.close()
       plt.figure(5)
       plt.plot(timebase, yValues, 'bo')
       t = np.linspace(timebase[0], timebase[-1])
       fitted = np.array(list(map(lambda x: modelFun(x, *params), t)))
       plt.plot(t, fitted, 'k')
       plt.title("Close the figure to continue")
       plt.show()
    
   buildUp_rate = np.zeros((len(PosStrain),len(PosStrain[0])))
   release_rate = np.zeros((len(PosStrain),len(PosStrain[0])))
   
   fitParamImg = np.zeros((len(PosStrain),len(PosStrain[0]),4))
   dt = info[1]['RepetitionTime'].value
   
   for ix in range(PosStrain.shape[0]):
      for iy in range(PosStrain.shape[1]):
           if mask[ix,iy] == True :

               xnew = np.arange(0, len(PosStrain[5,5,:])-1,0.1)
               x = np.arange(0,len(PosStrain[5,5,:]))#x = np.arange(0,len(StrainLine))
               x = x.astype(float)
               y = np.abs(PosStrain[ix,iy,:])
               # plt.figure(4)
               # plt.plot(y)
               if np.max(y) > 0:
                   f = interpolate.interp1d(x,y)
                   ynew = f(xnew)

                   firW_filt = signal.firwin(5,0.5)
                    
                   # Use lfilter to filter x with the FIR filter.
                   filtered_S = signal.lfilter(firW_filt, 1.0, ynew)
    
                   mx = np.max(filtered_S)
                   ind = np.argmax(filtered_S)

                   if np.min(filtered_S) > -0.05:
                       #building up strain-first half
                       e1_b = np.concatenate([filtered_S[0:ind], mx*np.ones((ind))],axis=0)
                       #release strain-2nd half
                       e1_r = np.concatenate([mx*np.ones((ind)), filtered_S[ind:-1]],axis=0)
                       e1_r = e1_r[::-1]
                                         
                       interp_f = len(ynew)/len(y)
                       xdata_b = (dt/interp_f)*np.arange(len(e1_b))
                       xdata_r = (dt/interp_f)*np.arange(len(e1_r))
                       plt.figure(6)
                       plt.plot(e1_r)
                                             
                       sigma_b = np.ones(len(xdata_b))
                       sigma_b[[0, -1]] = 0.01
                       
                       sigma_r = np.ones(len(xdata_r))
                       sigma_r[[0, -1]] = 0.01
                       
                       fitParams_bu,pcovariance = curve_fit(sigma_func, xdata_b, e1_b,p0 = (np.max(e1_b),np.min(e1_b),len(e1_b)/2,10),bounds=([0, -1,-30.,-30.], [10, 0.6,600.,400.]),method='trf',sigma=sigma_b)
                       fitParams_r,pcovarRel = curve_fit(sigma_func, xdata_r, e1_r,p0 = (np.max(e1_r),np.min(e1_r),len(e1_r)/2,10),bounds=([0, -1,-30.,-30.], [10, 0.6,800.,200.]),method='trf',sigma=sigma_r)                 
                       
                       y_fit = sigma_func(xdata_b,fitParams_bu[0],fitParams_bu[1],fitParams_bu[2],fitParams_bu[3])
                       
                       buildUp_rate[ix,iy] = (fitParams_bu[0]-fitParams_bu[1])/fitParams_bu[2]
                       release_rate[ix,iy] = (fitParams_r[0]-fitParams_r[1])/fitParams_r[2]                       
                       
                       fitParamImg[ix,iy,0]=fitParams_bu[0]
                       fitParamImg[ix,iy,1]=fitParams_bu[1]
                       fitParamImg[ix,iy,2]=fitParams_bu[2]
                       fitParamImg[ix,iy,3]=fitParams_bu[3]
                       
   #Fit whole ROI together (not pixelwise)
   mx = np.max(e1_Line)
   ind = np.argmax(e1_Line)
   #build-Up rate calculated from fitting the time decay of whole roi
   e_bUP_roi = np.concatenate([e1_Line[0:ind], mx*np.ones((ind))],axis=0)
   #release rate calculated from fitting the time decay of whole roi
   e_Rel_roi = np.concatenate([mx*np.ones((ind)), e1_Line[ind:-1]],axis=0)
   e_Rel_roi = e_Rel_roi[::-1]
      
   xdata_bUP = dt*np.arange(len(e_bUP_roi))
   xdata_Rel = dt*np.arange(len(e_Rel_roi))

   sigma_bUP = np.ones(len(xdata_bUP))
   sigma_bUP[[0, -1]] = 0.01
   sigma_Rel = np.ones(len(xdata_Rel))
   sigma_Rel[[0, -1]] = 0.01                
      
   params_bUP,pcovariance_S = curve_fit(sigma_func, xdata_bUP, e_bUP_roi,p0 = (np.max(e_bUP_roi),np.min(e_bUP_roi),len(e_bUP_roi)/2,10),bounds=([0,-1,-30.,-30.], [10,0.6,600.,400.]),method='trf',sigma=sigma_bUP)
   params_Rel,pcovariance_R = curve_fit(sigma_func, xdata_Rel, e_Rel_roi,p0 = (np.max(e_Rel_roi),np.min(e_Rel_roi),len(e_Rel_roi)/2,10),bounds=([0, -1,-30.,-30.], [10, 0.6,800.,200.]),method='trf',sigma=sigma_Rel)    
   
   showFit(xdata_bUP, e_bUP_roi, sigma_func, params_bUP)
   showFit(xdata_Rel, e_Rel_roi, sigma_func, params_Rel)
   
   buildUp_rate_roi = (params_bUP[0]-params_bUP[1])/params_bUP[2]
   Rel_rate_roi = (params_Rel[0]-params_Rel[1])/params_Rel[2]
    
   print("buildUp_rate_roi",np.round(buildUp_rate_roi,5))
   print(" Release_rate_roi", np.round(Rel_rate_roi,5))
   
   # print("fittedParams_Sum[0]roi",fittedParams_Sum[0])
   # print("fittedParams_Sum[1]roi",fittedParams_Sum[1])
   # print("fittedParams_Sum[2]roi",fittedParams_Sum[2])
   
   return buildUp_rate, fitParamImg
