#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for console
Created on Thu Feb 11 16:38:41 2021

@author: xenia
"""

from read_velocity import read_velocity
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')
%matplotlib Qt5
import numpy as np
from calc_disp import calc_disp
from mpl_toolkits.mplot3d import Axes3D

from calc_strain import calc_strain
from numpy import nonzero as nz

#dicom_dir = "/home/xenia/Documents/MATLAB/test_DICOM_2/005_FL_PC_3DIR/"
dicom_dir ="/home/xenia/Documents/MATLAB/DATA/test_dicom/007_fl_PC_3DIR_A/"
x,f2d,fx,fy,fz,mask,info = read_velocity(dicom_dir)


#plt.figure(0)
#plt.plot(x,f2d)

dispVxi, dispVyi, dispVzi = calc_disp(fx,fy,fz,info,mask)

#plt.figure(1)
#plt.imshow(dispVxi[:,:,10])

Eig_v = calc_strain(dispVxi,dispVyi,dispVzi,mask,fx,fy)

plt.figure(2)
plt.imshow(Eig_v[:,:,0,np.int(len(Eig_v[0][0][0])/2)])
plt.figure(2).suptitle('Princ.Strain(0) middle slice', fontsize=14)

# Summarize result, calculate texture parameters
from summarizeStrain import sum_strain
e1_Line, e1_max, e2_Line, e2_max  = sum_strain(Eig_v,mask)
plt.figure(3)
plt.plot(e1_Line)
from sigma_fit import calcRates

buildUp_rate,paramsImg = calcRates((Eig_v[:,:,0,:]),e1_Line,mask,info)

print("median buildUpPixelWise",np.median(buildUp_rate[nz(buildUp_rate)]))
p0=paramsImg[:,:,0]
p1=paramsImg[:,:,1]
p2=paramsImg[:,:,2]
print("p0",np.nanmedian(p0[mask]))
print("p1",np.nanmedian(p1[mask]))
print("p2",np.nanmedian(p2[mask]))
plt.figure(8)
plt.imshow(buildUp_rate)