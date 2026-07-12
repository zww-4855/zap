import copy

from copy import deepcopy
import dash_helper as dh
import build_R3 as br3

import numpy as np


def build_netT2_fromT3(g,o,v,t3):
    roovv = -0.250000000 * np.einsum("iklabc,jckl->ijab",t3,g[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijkacd,cdkb->ijab",t3,g[v,v,o,v],optimize="optimal")

    return roovv

def drive_R2_projection(W,o,v,C1,C2,T2,D3):
    # build the R2 residual associated with
    # 1) the square braket [T-6] term W*R2in the EOM energy expression
    # 2) the W*C1*T2 term in the EOM energy expression

    D2T2 = drive_eom_sqr_brak_t(W,eom_r2,o,v, D3)
    D2T2 += drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3)
    return  D2T2

def drive_R1_projection(W,o,v,C1,C2,T2,D3):
    # build the R1 residual associated with
    # 1) the square braket [T-6] term W*R2in the EOM energy expression
    # 2) the W*C1*T2 term in the EOM energy expression

    D1T1 = build_WnC2_to_R1(W,o,v,C2,D3)
    D1T1 += drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3)
    return  D1T1

def build_WnC1T2_to_R1(W,o,v,C1,T2,D3):
    # build the R1 residual associated with the W*C1*T2 term in the EOM energy expression
    C3_eff = br3.build_WnC1T2_to_T3(W,o,v,C1,T2,D3)

    # Now project WC3 -> C3
    D3C3 = br3.build_WT3_to_T3(W,o,v,C3_eff)
    D3C3 = br3.antisym_T3(copy.deepcopy(D3C3), 6, 6)

    # Now cap D3C3 with a T2^ to get the effective R1 
    # also, for testing purposes, cap this last expression 
    #  with R1^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
    D1C1 =  None

    return D1C1


def build_WnC2_to_R1(W,o,v,C2,D3):
    # build the R1 residual associated with the W*C2 term in the EOM energy expression
    C3_eff = build_sqr_brak_T3(W,o,v,C2,D3)

    # Now project WC3 -> C3
    D3C3 = br3.build_WT3_to_T3(W,o,v,C3_eff)
    D3T3 = br3.antisym_T3(copy.deepcopy(D3T3), 6, 6)

    # Now cap D3T3 with a T2^ to get the effective R1 
    # and cap with R1^ to verify the energy correction itself is correct
    # ------------ TO DO -----------------
    D1C1 =  None


    return D1C1



def build_sqr_brak_T3(W,o,v,eom_r2,D3):
    D3T3 = br3.build_T3_secondO_spin(W,o,v,eom_r2)
    D3T3 = br3.antisym_T3(copy.deepcopy(D3T3), 6, 6)
    eom_r3eff = D3T3 * D3
    return eom_r3eff

def build_WnC1T2_to_T3(W,o,v,C1,T2,D3):
    # build the second order T3 residual from W and C1T2
    D3C3_WC1T2 = br3.build_WnC1T2_to_T3(W,o,v,C1,T2)
    D3C3_WC1T2 = br3.antisym_T3(copy.deepcopy(D3C3_WC1T2), 6, 6)
    C3_eff =  D3C3_WC1T2 * D3
    return C3_eff

def drive_eom_sqr_brak_t(W,eom_r2,o,v, D3):
    # build effective T3 residual from W*R2 term in EOM energy expression
    eom_r3eff = build_sqr_brak_T3(W,o,v,eom_r2,D3)

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2 = build_netT2_fromT3(W,o,v,deepcopy(eom_r3eff))
    D2T2 = dh.antisym_T2(D2T2, 6, 6)

    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2, eom_r2.transpose(2,3,0,1))
    print(f" EOM [T]-like energy<D2T2|W|> = {sqr_brak_t_energy:.10f}")
    
    return D2T2


def drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3):
    #build effective T3 residual from W*C1*T2 term in EOM energy expression
    C3_eff = br3.build_WnC1T2_to_T3(W,o,v,C1,T2,D3)

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2_WC1T2 = build_netT2_fromT3(W,o,v,copy.deepcopy(C3_eff))
    D2T2_WC1T2 = dh.antisym_T2(copy.deepcopy( D2T2_WC1T2), 6, 6)

    
    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2_WC1T2, C2.transpose(2,3,0,1))
    print(f" EOM [T]-like energy<WC1T2|W|> = {sqr_brak_t_energy:.10f}")
    
    return  D2T2_WC1T2




