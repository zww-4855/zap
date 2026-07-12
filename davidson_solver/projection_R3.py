import copy

from copy import deepcopy
import dash_helper as dh
import build_R3 as br3

import numpy as np


def build_netT2_fromT3(g,o,v,t3):
    roovv = -0.250000000 * np.einsum("iklabc,jckl->ijab",t3,g[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijkacd,cdkb->ijab",t3,g[v,v,o,v],optimize="optimal")

    return roovv

def drive_eom_sqr_brak_t(W,eom_r2,o,v, D3):
    # first build fourth order T3 residual,
    # antisymmeterize it
    # then contract against D3 to form T3 triples tensor.
    D3T3 = br3.build_T3_secondO_spin(W,o,v,eom_r2)
    D3T3 = br3.antisym_T3(copy.deepcopy(D3T3), 6, 6)
    eom_r3eff = D3T3 * D3

    # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2 = build_netT2_fromT3(W,o,v,deepcopy(eom_r3eff))
    D2T2 = dh.antisym_T2(D2T2, 6, 6)

    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2, eom_r2.transpose(2,3,0,1))
    print(f" EOM [T]-like energy<D2T2|W|> = {sqr_brak_t_energy:.10f}")
    
    return D2T2


def drive_eom_WnC1T2_to_T3(W,o,v,C1,C2,T2,D3):
    # build the second order T3 residual from W and C1T2
    D3C3_WC1T2 = br3.build_WnC1T2_to_T3(W,o,v,C1,T2)
    D3C3_WC1T2 = br3.antisym_T3(copy.deepcopy(D3C3_WC1T2), 6, 6)
    C3_eff =  D3C3_WC1T2 * D3


       # project the triples tensor into the doubles space to form a second order T2 residual
    D2T2_WC1T2 = build_netT2_fromT3(W,o,v,copy.deepcopy(C3_eff))
    D2T2_WC1T2 = dh.antisym_T2(copy.deepcopy( D2T2_WC1T2), 6, 6)

    
    sqr_brak_t_energy = 0.25*np.einsum('jiab,abji',D2T2_WC1T2, C2.transpose(2,3,0,1))
    print(f" EOM [T]-like energy<WC1T2|W|> = {sqr_brak_t_energy:.10f}")
    
    return  D2T2_WC1T2


