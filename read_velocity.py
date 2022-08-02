
# -*- coding: utf-8 -*-
"""
Purpose: Calculate motion field and visualize velocity maps
Xeni Deligianni- 2021-Routine to calculate Strain Maps from slices of PC dicom-data

nr_gre: number of gre that is used for ROI segmentation, or if there is no extra gre image, 0 
"""

#from matplotlib import pyplot, cm
#from matplotlib.pyplot import imshow
#import in command line: %matplotlib Qt5
#from IPython import get_ipython
#ipython = get_ipython()
# ipython.magic("matplotlib Qt5")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings

from pathlib import Path
import os
from dicomUtils.dicom3D import load3dDicom
from roipoly import RoiPoly
import numpy as np
from mrprot import read_siemens_prot
from scipy import ndimage, misc
import math 

def read_velocity(dirname):
    ##################
    
    #dirname="/home/xenia/Documents/MATLAB/test_dicom/007_fl_PC_3DIR_A/"
    fpath = Path(dirname)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    head_tail = os.path.split(fpath) 
     
    # print head and tail of the specified path 
    print("Head of '% s:'" % fpath, head_tail[0]) 
    print("Tail of '% s:'" % fpath, head_tail[1], "\n") 
    
    filename_mag=head_tail[1]
    str_nr=float(filename_mag[1:3])
    filename_p = filename_mag[3:(len(filename_mag))]
    s=""
    
    filename_ph=s.join([filename_p,"_P"])
    if (str_nr+2) <= 9:
        fpath_px=s.join([head_tail[0],"/00",str(int(str_nr+2)),filename_ph,"/"])
    else:
        fpath_px=s.join([head_tail[0],"/0",str(int(str_nr+2)),filename_ph,"/"])
    if (str_nr+4) <= 9:
        fpath_py=s.join([head_tail[0],"/00",str(int(str_nr+4)),filename_ph,"/"])
    else:
        fpath_py=s.join([head_tail[0],"/0",str(int(str_nr+4)),filename_ph,"/"])
    if (str_nr+6) <= 9:
        fpath_pz=s.join([head_tail[0],"/00",str(int(str_nr+6)),filename_ph,"/"])
    else:
        fpath_pz=s.join([head_tail[0],"/0",str(int(str_nr+6)),filename_ph,"/"])
    
    #Read phase contrast data
    magn_img, info = load3dDicom(fpath)
    phase_x_img, info_x = load3dDicom(fpath_px)
    phase_y_img, info_y = load3dDicom(fpath_py)
    phase_z_img, info_z = load3dDicom(fpath_pz)
    
    dim0 = magn_img.shape[0]
    dim1 = magn_img.shape[1] #magnitude image
    nr_gre=int(input("GRE series number(0 for none): "))
    #Read gre slice for segmentation if it exists
    #if not segment on
    
    if nr_gre > 9:
        fpath_gre=s.join([head_tail[0],"/0",str(nr_gre),"_gre_ROI","/"]) 
        gre_img, info = load3dDicom(fpath_gre)
    elif nr_gre > 1:
        fpath_gre=s.join([head_tail[0],"/00",str(nr_gre),"_gre_ROI","/"])
        gre_img, info = load3dDicom(fpath_gre)
        
    #Parameters from siemens dicom header(to be replaced for release?)
    prot = read_siemens_prot(info[0])
    
    venc=prot['sAngio']['sFlowArray']['asElm'][0]['nVelocity']
    acqTimeWindow = prot['sPhysioImaging']['sPhysioExt']['lScanWindow']
    trigDelay = prot['sPhysioImaging']['sPhysioExt']['lTriggerDelay']/1000
    nPhases = int(info_x[1]['CardiacNumberOfImages'].value)
    
    #Hard-coded parameters
    nThresh = 10# image noise threshold
        
    #segment ROI
    plt.imshow(magn_img[:,:,10])
    plt.title("left click: line segment         right click or double click: close region")
    my_roi=RoiPoly(color='r')
    if nr_gre==0:
        mask = my_roi.get_mask(magn_img[:,:,10])
    else:
        mask = my_roi.get_mask(gre_img)
    
     
    # Adapt range of phase values & median filtering
    flow_x=np.zeros((len(magn_img),len(magn_img[0]),len(magn_img[0][0])))
    flow_y=np.zeros((len(magn_img),len(magn_img[0]),len(magn_img[0][0])))
    flow_z=np.zeros((len(magn_img),len(magn_img[0]),len(magn_img[0][0])))
    for i_sl in range(nPhases):
        flow_x[:,:,i_sl] = (magn_img[:,:,i_sl] > nThresh)*mask*(np.subtract(phase_x_img[:,:,i_sl],2048))/2048
        flow_y[:,:,i_sl] = (magn_img[:,:,i_sl] > nThresh)*mask*(np.subtract(phase_y_img[:,:,i_sl],2048))/2048
        flow_z[:,:,i_sl] = (magn_img[:,:,i_sl] > nThresh)*mask*(np.subtract(phase_z_img[:,:,i_sl],2048))/2048
        #median filtering & shading correction
        flow_x[:,:,i_sl] = ndimage.median_filter(flow_x[:,:,i_sl]*venc, size=5,mode='mirror')
        flow_y[:,:,i_sl] = ndimage.median_filter(flow_y[:,:,i_sl]*venc, size=5,mode='mirror')
        flow_z[:,:,i_sl] = ndimage.median_filter(flow_z[:,:,i_sl]*venc, size=5,mode='mirror')  
        
    shade_x=np.mean(flow_x,axis=2)
    shade_y=np.mean(flow_y,axis=2)
    shade_z=np.mean(flow_z,axis=2)
    
    for i_sl in range(nPhases):
        flow_x[:,:,i_sl] = np.subtract(flow_x[:,:,i_sl],shade_x)
        flow_y[:,:,i_sl] = np.subtract(flow_y[:,:,i_sl],shade_y)
        flow_z[:,:,i_sl] = np.subtract(flow_z[:,:,i_sl],shade_z)  
        
    
    temp_flow_2D=np.zeros((len(magn_img),len(magn_img[0]),len(magn_img[0][0])))
    flow_2D=np.zeros(len(magn_img[0][0])) 
                  
    for iP in range(nPhases):
        temp_flow_2D[:,:,iP] = np.sqrt(pow(flow_x[:,:,iP],2)+pow(flow_y[:,:,iP],2))
        temp_img= temp_flow_2D[:,:,iP]
        flow_2D[iP]=np.median(temp_img[mask])#was median originally
    
    x=list(range(0, nPhases)) 
    plt.figure(0)
    plt.plot(x,flow_2D,color='r')
    plt.figure(0).suptitle('Flow_2D-median of ROI', fontsize=14)
    plt.show()
    return x,flow_2D,flow_x,flow_y,flow_z,mask,info
    
