# -*- coding: utf-8 -*-
# Import Packages
import numpy as np
import pandas as pd
import os

def Desaggregation_BaU_NBS(PathProject):
    # Funtion Lambda
    Sigmoid_Desaggregation = lambda Wmax, Wo, r, t: Wmax/(1 + (((Wmax/Wo) - 1)*np.exp(-t*r)))

    NameCol     = ['AWY (m3)','Wsed (Ton)','WN (Kg)','WP (kg)','BF (m3)','WC (Ton)']
    Data        = pd.read_csv(os.path.join(PathProject,'01-INPUTS_InVEST.csv'),usecols=NameCol)
    NBS         = pd.read_csv(os.path.join(PathProject,'01-INPUTS_NBS.csv')).values[:,1:]
    Time        = pd.read_csv(os.path.join(PathProject,'01-INPUTS_Time.csv')).values[0][0]
    
    # Check AWY = 0  -> 0.0031536 m3 = 0.1 l/s
    id = Data['AWY (m3)'] == 0
    Data['AWY (m3)'].iloc[id] = 0.0031536
    
    # Control Error - Zero - OJO !!!!!
    id = sum(NBS) > 0
    NBS = NBS[:,id]
    
    '''
    Current-BaU
    '''
    nn = nn = np.shape(Data)[1]
    Results_BaU = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)
    r = -1*np.log(0.000000001)/Time
    t = np.arange(0,Time + 1)
    for i in range(0,6):
        Results_BaU[NameCol[i]] = Sigmoid_Desaggregation(Data[NameCol[i]][1], Data[NameCol[i]][0], r, t)

    '''
    BaU-NBS
    '''
    # Estimation Time NBS
    n = np.size(NBS[0,2:])
    t_NBS = np.empty([n,1])
    p_NBS = np.empty([n,1])
    for i in range(0,n):
        t_NBS[i] = np.nansum(NBS[:,0]* NBS[:,i+2])/np.nansum(NBS[:,i+2])
        p_NBS[i] = np.nansum(NBS[:,1]* NBS[:,i+2])/np.nansum(NBS[:,i+2])

    # Desaggregation
    Results_NBS = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)

    # Estimation Diff
    [f,c]       = Data.shape
    Data1       = Data[2:].values
    Data1[0,:]  = Data1[0,:] - Data.loc[1].values
    Tmp         = np.nancumsum(Data1,0)
    for i in range(1,f-2):
        Data1[i,:]   = Data1[i,:] - Data.loc[1].values - Tmp[i-1,:]
        Tmp = np.nancumsum(Data1,0)

    for i in range(0,nn):
        Tmp = np.zeros((Time+1,n))
        for j in range(0,n):
            t    = np.arange(0, Time + 1 - (j+1))
            Wmax = Data1[j,i]
            tmax = t_NBS[j][0]
            Wo   = p_NBS[j][0]*Data1[j,i]*0.01
            r    = -1*np.log(0.000000001)/tmax

            #print(Wmax,'|', Wo,'|', r)
            Tmp[(j+1):,j] = Sigmoid_Desaggregation(Wmax, Wo, r, t)
        
        Tmp[np.isnan(Tmp)] = 0
        Results_NBS[NameCol[i]] = np.sum(Tmp,1) + Results_BaU[NameCol[i]].values
    
    # Correct Negative Values
    Posi = Results_NBS.index.values
    for i in range(0,nn):
        if np.sum(Results_NBS.iloc[:, i] < 0) > 0:
            TmpBaU = Results_BaU.iloc[:, i]
            TmpNBS = Results_NBS.iloc[:, i]

            n = np.max(Posi[TmpNBS < 0]) + 1
            V = (TmpBaU[n] - TmpNBS[n])/TmpBaU[n]
            for j in range(1,n):
                TmpNBS[j] = TmpBaU[j]*V
            Results_NBS.iloc[:, i] = TmpNBS
        else:
            print('No Correct -> '+NameCol[i])
            
    '''    
    Save Data
    '''
    Results_BaU.to_csv(os.path.join(PathProject,'02-OUTPUTS_BaU.csv'), index_label='Time')
    Results_NBS.to_csv(os.path.join(PathProject,'02-OUTPUTS_NBS.csv'), index_label='Time')

# terter
#PathProject = r'Z:\Box Sync\01-TNC-ThinkPad-P51\28-Project-WaterFund_App\02-Productos-Intermedios\Python_Convolution\Project'
#Desaggregation_BaU_NBS(PathProject)
