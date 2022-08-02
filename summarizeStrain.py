#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:31:50 2021
Summarize the maps of strain
Calculate time evolution
@author: xenia
"""
import numpy as np
from numpy import nonzero as nz


def sum_strain(Eig_v,mask):

    nPhases= Eig_v.shape[3]
    e1_Line = np.zeros(nPhases)
    e2_Line = np.zeros(nPhases)
    
    for i_sl in range(1,nPhases):        
        E1= Eig_v[:,:,0,i_sl]
        E2= Eig_v[:,:,1,i_sl]
        
        e1_Line[i_sl] = (np.nanmedian(E1[nz(E1)]))
        e2_Line[i_sl] = np.absolute(np.nanmedian(E2[nz(E2)]))
    
    e1_max = np.round(np.nanmax(e1_Line),3)# First strain
    e2_max = np.round(np.nanmax(e2_Line),3)# Second strain
    
    return e1_Line, e1_max, e2_Line, e2_max