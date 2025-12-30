#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:57:37 2023

@author: solar
"""
import numpy as np
def mbe(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss


def mae(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss

def rmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)


def rmbe(true, pred):
    mbe_loss = np.mean(pred  - true)
    return mbe_loss/ true.mean() * 100


def rmae(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss/ true.mean() * 100

def rrmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)  / true.mean() * 100


#------------------------------------
def ecdf(x): #CDF empirica
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def KSI_OVER(Xval,Xest,CDF=0): #CDF es para ver las series
    #---6/07/2015
    #---25/09/2018 - version Python
    #---30/04/2019 - version con forma mas eficiente para bajar el tiempo
    #--- Inicialización
    sVAL = len(Xval)
    sEST = len(Xest)
    if sVAL!=sEST: print('longitudes diferentes, valores relativos no validos')
    Vc = 1.63/np.sqrt(sVAL)
    
    [xCDFest, CDFest]=ecdf(Xest)
    [xCDFval, CDFval]=ecdf(Xval)
    
    xCDF_tot=np.unique(np.concatenate((Xval, Xest))) #concantenate: pega los vectores
    
    CDFval_tot = np.interp(xCDF_tot, xCDFval, CDFval) # interpolo para que sean vectores iguales
    CDFest_tot = np.interp(xCDF_tot, xCDFest, CDFest) # interpolo para que sean vectores iguales
    
    Xmax=max(xCDF_tot); Xmin=min(xCDF_tot)
        # --- Dn y On:
    Dn = abs(CDFval_tot - CDFest_tot)
    On = (Dn - Vc)*(Dn > Vc)
    #---indicadores:
    KSI  = np.trapz(Dn, xCDF_tot)
    OVER = np.trapz(On, xCDF_tot)
    
    # --- Relativos
    rKSI  = KSI/ (Vc*(Xmax - Xmin))
    rOVER = OVER/(Vc*(Xmax - Xmin))
    if CDF ==1: 
        return KSI, OVER, rKSI, rOVER, xCDF_tot, CDFval_tot, CDFest_tot, Dn, On, Vc
    else:
        #return KSI, OVER, rKSI, rOVER
        return KSI