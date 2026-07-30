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
    
    
    D2T2a = build_WnC1T2_to_R2(W,o,v,C1,C2,T2,D3) # drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3)
    e = 0.25*np.einsum('jiab,abji',D2T2a, C2.transpose(2,3,0,1))

    print(f" Final R2-EOM check for Diagram C (wnr1t2) = {e:.10f}\n\n\n" )
    #sys.exit()
    return  D2T2+D2T2a

def build_WnC1T2_to_R2(W,o,v,C1,C2,T2,D3):
    # build the R2 residual associated with the W*C1*T2 term in the EOM energy expression
    C3_eff = build_WnC1T2_to_T3(W,o,v,C1,T2,D3)

    # Now project WC3 -> C3
    #D3C3 = br3.build_WT3_to_T3(W,o,v,C3_eff)
    #D3C3 = br3.antisym_T3(copy.deepcopy(D3C3), 6, 6)

    # Now cap D3C3 with a T2^ to get the effective R2 
    # also, for testing purposes, cap this last expression 
    #  with R2^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
    D2C2 =  br3.build_netT2_fromT3(W,o,v,C3_eff)
    D2C2 = br3.antisym_T2(copy.deepcopy(D2C2), 6, 6)
    return D2C2

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



def drive_R1_projection(W,o,v,C1,C2,T2,D3):
    # build the R1 residual associated with
    # 1) the square braket [T-6] term W*R2in the EOM energy expression
    # 2) the W*C1*T2 term in the EOM energy expression

    D1R1 = build_WnC2_to_R1(W,o,v,T2,C2,D3)
    tmp_E1 = np.einsum("ia,jb->",D1R1,C1.transpose(1,0),optimize="optimal")
    print(f" Final R1-EOM check for Q3(WnC2) -> C1 Diagram B = {tmp_E1:.10f}")

    D1T1a = build_WnC1T2_to_R1(W,o,v,C1,C2,T2,D3)
    tmp_E1 = np.einsum("ia,jb->",D1T1a,C1.transpose(1,0),optimize="optimal")
    print(f" Final R1-EOM check for Q3(WnC1T2) -> C1 Diagram A = {tmp_E1:.10f}")
    sys.exit()
    return  D1R1+D1T1a

def build_WnC1T2_to_R1(W,o,v,C1,C2,T2,D3):
    # build the R1 residual associated with the W*C1*T2 term in the EOM energy expression
    #C3_eff = build_WnC1T2_to_T3(W,o,v,C1,T2,D3)
    D3C3_WC1T2 = br3.build_WnC1T2_to_T3(W,o,v,C1,T2)
    C3_eff =  copy.deepcopy(D3C3_WC1T2) * D3
    D1R1_eff = br3.build_T2dagWT3_to_D1R1eff(T2, W, C3_eff, o, v)

    # Now project WC3 -> C3
    #D3C3 = br3.build_WT3_to_T3(W,o,v,C3_eff)
    #D3C3 = br3.antisym_T3(copy.deepcopy(D3C3), 6, 6)

    # Now cap D3C3 with a T2^ to get the effective R1 
    # also, for testing purposes, cap this last expression 
    #  with R1^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
   # D1C1 =  br3.build_R3eff_to_R1(o,v,D3C3,T2)
    return D1R1_eff


def build_WnC2_to_R1(W,o,v,T2,C2,D3):
    # build the R1 residual associated with the W*C2 term in the EOM energy expression
    C3_eff = build_sqr_brak_T3(W,o,v,C2,D3) 

    D1C1 = br3.build_T2dagWT3_to_D1R1eff(T2, W, C3_eff, o, v)


    # Now project WC3 -> C3
    #D3C3 = br3.build_WT3_to_T3(W,o,v,C3_eff)
    #D3C3 = br3.antisym_T3(copy.deepcopy(D3C3), 6, 6)

    # Now cap D3C3 with a T2^ to get the effective R1 
    # and cap with R1^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
    #D1C1 = br3.build_R3eff_to_R1(o,v,D3C3,T2)


    return D1C1



def build_sqr_brak_T3(W,o,v,eom_r2,D3):
    D3T3 = br3.build_T3_secondO_spin(W,o,v,eom_r2)
    D3T3 = br3.antisym_T3(copy.deepcopy(D3T3), 6, 6)
    eom_r3eff = D3T3 * D3

    #tmp= (1/36.0)*np.einsum("ijkabc,abcijk->",eom_r3eff,D3T3.transpose(3,4,5,0,1,2),optimize="optimal")
    #print("EOM [T-6] sqrbrakT diagram D: ",tmp)
    return eom_r3eff
  


def build_WnC1T2_to_T3(W,o,v,C1,T2,D3):
    # build the second order T3 residual from W and C1T2
    D3C3_WC1T2 = br3.build_WnC1T2_to_T3(W,o,v,C1,T2)
    D3C3_WC1T2 = br3.antisym_T3(copy.deepcopy(D3C3_WC1T2), 6, 6)
    C3_eff =  copy.deepcopy(D3C3_WC1T2) * D3
    tmp= (1/36.0)*np.einsum("ijkabc,abcijk->",C3_eff,D3C3_WC1T2.transpose(3,4,5,0,1,2),optimize="optimal")
    print("EOM [T-6] energy correction diagram A:",tmp)
    #sys.exit()
    return C3_eff

def drive_eom_sqr_brak_t(W,eom_r2,o,v, D3):
    # build effective T3 residual from W*R2 term in EOM energy expression
    eom_r3eff = build_sqr_brak_T3(W,o,v,eom_r2,D3)

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2 = br3.build_netT2_fromT3(W,o,v,deepcopy(eom_r3eff))
    D2T2 = br3.antisym_T2(D2T2, 6, 6)

    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2, eom_r2.transpose(2,3,0,1))
    print(f"EOM [T-6] sqrbrakT diagram D: (checked using Q2proj) {sqr_brak_t_energy:.14f}")
    
    return D2T2


def drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3):
    #build effective T3 residual from W*C1*T2 term in EOM energy expression
    C3_eff = br3.build_WnC1T2_to_T3(W,o,v,C1,T2)
    D3T3 = br3.antisym_T3(copy.deepcopy(C3_eff), 6, 6)
    C3 = D3T3 * D3

    # Now project WC3 -> C3
    D3C3 = br3.build_WT3_to_T3(W,o,v,C3)
    D3C3 = br3.antisym_T3(copy.deepcopy(D3C3), 6, 6)

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2_WC1T2 = br3.build_netT2_fromT3(W,o,v,copy.deepcopy(C3))
    D2T2_WC1T2 = br3.antisym_T2(copy.deepcopy( D2T2_WC1T2), 6, 6)

    
    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2_WC1T2, C2.transpose(2,3,0,1))
    print(f" WnC1T2_to_T3 EOM [T]-like energy<WC1T2|W|> = {sqr_brak_t_energy:.10f}")
    print()
    
    return  D2T2_WC1T2




