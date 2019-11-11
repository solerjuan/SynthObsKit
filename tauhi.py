#
# 
# Copyright (C) 2013-2018 Juan Diego Soler, Patrick Hennebelle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits

from scipy import misc
from scipy.ndimage import rotate
from scipy.signal import gaussian

from astropy.io import ascii
from astropy.table import Table
from astropy import constants as const
from astropy import units as u

#def tauhi(dens, temperature, vlos, pos, vsize=100, dv=1.0, selfabsorption=True, dens_thres=5.0E2, dens_max=1.0E4, temp_thres=5.0E2):
def tauhi(dens, temperature, vlos, pos, vsize=100, dv=1.0, selfabsorption=True, mincodens=500.0, maxcot=50.0, minhit=0.0, maxcnmt=200.0, minwnmt=5000., maxwnmt=8500.):
   # Calculate synthetic HI emission
   #
   # INPUTS
   # dens - vector of densities [(1/cm)**3]
   # temperature - vector of temperatures [K]
   # vlos - vector of line of sight velocities [cm/s]
   #
   # OUTPUTS
   # velvec - velocity vector  
   # tau - optical depth
   # emiss_all -
   # emiss_wnm -
   # emiss_cnm -

   nb=np.size(dens)
   dens_co=dens.copy()
   dens_wnm=dens.copy()
   dens_cnm=dens.copy()
   dens_all=dens.copy() 

   #here we very roughly set that if the density is larger than 500 cm**-3 the gas is molecular (should be improved a lot)
   selpix=np.logical_or(temperature > maxcot, dens < mincodens).nonzero()
   if(np.size(selpix) > 0):
      dens_co[selpix]=0.
 
   selpix=np.logical_or(temperature < minhit, temperature > maxcnmt).nonzero()
   if(np.size(selpix) > 0):
      dens_cnm[selpix]=0.
   
   selpix=np.logical_or(temperature < minwnmt, temperature > maxwnmt).nonzero()
   if(np.size(selpix) > 0):
      dens_wnm[selpix]=0.

   selpix=np.logical_or(temperature < minhit, temperature > maxwnmt).nonzero() 
   if(np.size(selpix) > 0):
      dens_all[selpix]=0.

   # Initialize
   velvec=(np.arange(vsize, dtype='float')-float(vsize)/2.)*dv*1.0E5 # cm/s
   tau_all=np.zeros(vsize)
   tau_cnm=np.zeros(vsize)
   tau_co=np.zeros(vsize)

   emiss_wnm=np.zeros(vsize)        
   emiss_cnm=np.zeros(vsize)
   emiss_all=np.zeros(vsize)
   emiss_co =np.zeros(vsize)

   emiss_wnmTHIN=np.zeros(vsize)
   emiss_cnmTHIN=np.zeros(vsize)
   emiss_allTHIN=np.zeros(vsize)
   emiss_coTHIN =np.zeros(vsize)

   # Parameters for the HI transtion
   l_doppl=2.0*const.k_B.cgs.value/const.m_p.cgs.value
   coef_abso=np.float64(05.49E-14) #Absorption coefficient [cgs] (Spitzer 1978 p43)
    
   l_doppl_dense=l_doppl/28. # about the mass of CO

   b_wnm=np.sqrt(2*const.k_B.cgs.value*temperature/const.m_p.cgs.value)
   b_cnm=np.sqrt(2*const.k_B.cgs.value*temperature/const.m_p.cgs.value)   
   b_all=np.sqrt(2*const.k_B.cgs.value*temperature/const.m_p.cgs.value)
   b_co =np.sqrt(2*const.k_B.cgs.value*temperature/(const.m_p.cgs.value*28.))

   dx=np.roll(pos,-1)-pos
   dx[nb-1]=pos[nb-1]-pos[nb-2]

   for i in range(0,nb):
      p_wnm=np.exp(-(velvec-vlos[i])**2/b_wnm[i]**2)/(np.sqrt(np.pi)*b_wnm[i])
      p_cnm=np.exp(-(velvec-vlos[i])**2/b_cnm[i]**2)/(np.sqrt(np.pi)*b_cnm[i])
      p_all=np.exp(-(velvec-vlos[i])**2/b_all[i]**2)/(np.sqrt(np.pi)*b_all[i])
      p_co =np.exp(-(velvec-vlos[i])**2/b_co[i]**2 )/(np.sqrt(np.pi)*b_co[i] )

      tau_cell_wnm=dens_wnm[i]*dx[i]*p_cnm*coef_abso/temperature[i]
      tau_cell_cnm=dens_cnm[i]*dx[i]*p_cnm*coef_abso/temperature[i]
      tau_cell_all=dens[i]*dx[i]*p_cnm*coef_abso/temperature[i]
      tau_cell_co=dens_co[i]*dx[i]*p_co*coef_abso/temperature[i]

      if (selfabsorption):
         absorp_all=np.exp(-tau_all)
         absorp_co =np.exp(-tau_co) 
      else:
         absorp_all=1.
         absorp_co =1.

      emiss_wnmTHIN=emiss_wnmTHIN.copy()+temperature[i]*tau_cell_wnm
      emiss_cnmTHIN=emiss_cnmTHIN.copy()+temperature[i]*tau_cell_cnm
      emiss_allTHIN=emiss_allTHIN.copy()+temperature[i]*tau_cell_all
      emiss_coTHIN =emiss_coTHIN.copy() +temperature[i]*tau_cell_co

      emiss_wnm=emiss_wnm.copy()+temperature[i]*(1.0-np.exp(-tau_cell_wnm))*absorp_all
      emiss_cnm=emiss_cnm.copy()+temperature[i]*(1.0-np.exp(-tau_cell_cnm))*absorp_all
      emiss_all=emiss_all.copy()+temperature[i]*(1.0-np.exp(-tau_cell_all))*absorp_all
      emiss_co =emiss_co.copy() +temperature[i]*(1.0-np.exp(-tau_cell_co ))*absorp_co

      tau_all=tau_all.copy()+tau_cell_all
 
   velvec=velvec.copy()/1.0E5   # velocity vector in km/s 

   return velvec, tau_all, emiss_all, emiss_wnm, emiss_cnm, tau_co, emiss_co


