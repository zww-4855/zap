import copy

from copy import deepcopy
import dash_helper as dh
import build_R3 as br3

import numpy as np

def drive_R2_projection(W,o,v,C1,C2,T2,D3):
    # build the R2 residual associated with
    # 1) the square braket [T-6] term W*R2in the EOM energy expression
    # 2) the W*C1*T2 term in the EOM energy expression

    D2T2 = build_WnC2_to_R2(W,o,v,T2,C2,D3) #drive_eom_sqr_brak_t(W,C2,o,v, D3)
    sqrbrak_E =0.25*np.einsum('jiab,abji',D2T2, C2.transpose(2,3,0,1))
    print(f" Final R2-EOM check for Diagram D (sqrbrak) = {sqrbrak_E:.10f}")
    
    #sys.exit()
    return  D2T2



def build_WnC2_to_R2(W,o,v,T2,C2,D3):
    # build the R2 residual associated with the W*C2 term in the EOM energy expression
    C3_eff = build_sqr_brak_T3(W,o,v,C2,D3)


    
    # Now cap D3C3 with a T2^ to get the effective R2 
    # also, for testing purposes, cap this last expression 
    #  with R2^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
    D2C2 =  br3.build_netT2_fromT3(W,o,v,C3_eff)
    D2C2 = br3.antisym_T2(copy.deepcopy(D2C2), 6, 6)
    return D2C2




def build_sqr_brak_T3(W,o,v,eom_r2,D3):
    D3T3 = br3.build_T3_secondO_spin(W,o,v,eom_r2)
    D3T3 = br3.antisym_T3(copy.deepcopy(D3T3), 6, 6)
    eom_r3eff = D3T3 * D3

    #tmp= (1/36.0)*np.einsum("ijkabc,abcijk->",eom_r3eff,D3T3.transpose(3,4,5,0,1,2),optimize="optimal")
    #print("EOM [T-6] sqrbrakT diagram D: ",tmp)
    return eom_r3eff
  

def drive_eom_sqr_brak_t(W,eom_r2,o,v, D3):
    # build effective T3 residual from W*R2 term in EOM energy expression
    eom_r3eff = build_sqr_brak_T3(W,o,v,eom_r2,D3)

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2 = br3.build_netT2_fromT3(W,o,v,deepcopy(eom_r3eff))
    D2T2 = br3.antisym_T2(D2T2, 6, 6)

    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2, eom_r2.transpose(2,3,0,1))
    print(f"EOM [T-6] sqrbrakT diagram D: (checked using Q2proj) {sqr_brak_t_energy:.14f}")
    
    return D2T2



