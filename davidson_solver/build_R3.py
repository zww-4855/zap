import numpy as np

def antisym_T3(Rooovvv, nocc, nvir):
    # antisymmetrize the residual

    nocc=Rooovvv.shape[0]
    nvir=Rooovvv.shape[3]
    Rooovvv_anti = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir))
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkbac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkbca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkcab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjbac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjbca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjcab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikbac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikbca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikcab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikcba", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkiabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkiacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkibac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkibca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkicab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkicba", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijbac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijbca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijcab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjiabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjiacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjibac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjibca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjicab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjicba", Rooovvv)
    return Rooovvv_anti

def build_T3_secondO_spin(g,o,v,t2):
    rooovvv = -0.250000000 * np.einsum("ilab,jklc->ijkabc",t2,g[o,o,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",t2,g[o,v,v,v],optimize="optimal")
    return rooovvv


def build_eom_sqrbrakT_resid(W,T2,C1,C2,o,v):
    # return antisymmetrized terms defininig D3C3

    D3C3_WR1T2 = build_Q3_WR1T2(W,C1,T2,o,v)
    D3C3_WC2   = build_Q3_WnC2(W,C2,o,v)

    return D3C3_WR1T2, D3C3_WC2


def build_Q3_WR1T2(W,C1,T2,o,v):
    no = np.shape(C1)[0]
    nv = np.shape(C1)[1]
    d3c3 = np.zeros((no,no,no,nv,nv,nv))
    d3c3 += 0.250000000 * np.einsum("ilab,mc,jklm->ijkabc",T2,C1,W[o,o,o,o],optimize="optimal")
    d3c3 += 0.500000000 * np.einsum("ilab,jd,kdlc->ijkabc",T2,C1,W[o,v,o,v],optimize="optimal")
    d3c3 += 0.500000000 * np.einsum("ijad,lb,kdlc->ijkabc",T2,C1,W[o,v,o,v],optimize="optimal")
    d3c3 += 0.250000000 * np.einsum("ijad,ke,debc->ijkabc",T2,C1,W[v,v,v,v],optimize="optimal")

    fin_d3c3 = antisym_T3(d3c3, no, nv)
    return fin_d3c3


def build_Q3_WnC2(W,C2,o,v):
    no = np.shape(C2)[0]
    nv = np.shape(C2)[2]
    d3c3 = build_T3_secondO_spin(W,o,v,C2)
    fin_d3c3 = antisym_T3(d3c3,no,nv)
    return fin_d3c3



### WILL NEED THIS FOR LITERALLY ALL OF THE REMAINING EOM TERMS
def build_WT3_to_T3(W,o,v,T3):
    rooovvv = 0.041666667 * np.einsum("ilmabc,jklm->ijkabc",T3,W[o,o,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijlabd,kdlc->ijkabc",T3,W[o,v,o,v],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("ijkade,debc->ijkabc",T3,W[v,v,v,v],optimize="optimal")
    return rooovvv


def build_WnC1T2_to_T3(W,o,v,C1,T2):
    rooovvv = -0.250000000 * np.einsum("ijlm,la,kmbc->ijkabc",W[o,o,o,o],C1,T2,optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("idla,lb,jkcd->ijkabc",W[o,v,o,v],C1,T2,optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("idla,jd,klbc->ijkabc",W[o,v,o,v],C1,T2,optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("deab,id,jkce->ijkabc",W[v,v,v,v],C1,T2,optimize="optimal")
    return rooovvv

