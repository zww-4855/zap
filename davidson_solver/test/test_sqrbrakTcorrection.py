import pytest
#from davidson_solver.davidson import davidson
from ..davidson import davidson
from ..harvest_base_data import *
from ..dash_helper import *
from ..build_R3 import * 
from ..projection_R3 import * 
import copy
import numpy as np


def test_sqrbrakT_raw_energy_spectra(): 

    np.random.seed(7)
    # Load matrix representing CH+ molecule
    re_path = "/Users/zwu/Desktop/work/iterative_EOM/CH_Hbars/re_full.txt"
    import read_info as ri
    Hbar = ri.read_Hbar_matrix_from_txt(re_path, dim=117, dtype=float)
    E_uccsd = -37.9174827003253  # example ground-state UCCSD energy
    Hbar_shifted = ri.subtract_ground_state_energy_from_diagonal(
        Hbar,
        E_uccsd,
    )
    Hbar = Hbar_shifted

    # ------------------------------------------------------------
    # Davidson diagonalization
    # ------------------------------------------------------------
    evals_davidson, evecs_davidson, res = davidson(
        Hbar,
        n_roots=15,
        max_iter=10000,
        max_subspace=120,
        tol=1e-13,
        diag=np.diag(Hbar),
        verbose=True,
    )



    inputs = EOMUCCSDInputData(
        fock_file=(
            "/Users/zwu/Desktop/work/bkup/zap/outputs/"
            "CH+re/CH+ Full operator/fock.txt"
        ),
        tei_file=(
            "/Users/zwu/Desktop/work/bkup/zap/outputs/"
            "CH+re/CH+ Full operator/two_elec.txt"
        ),
        amplitudes_file=(
            "/Users/zwu/Desktop/work/bkup/zap/outputs/"
            "CH+re/CH+ Full operator/t1_t2.txt"
        ),
    )

    inputs.print_summary()
    sqrbrkT_values=[]
    for root_num in range(0,15):
        root = root_num
        omega = float(np.real(evals_davidson[root]))
        D3 = build_d3_for_root(
            inputs,
            omega=omega,
            level_shift=0.0,
        )
        
        r1 = expand_r1(evecs_davidson[:18, root],inputs.nocc, inputs.nvirt)
        r1, r2 = parse_davidson_eigenvector_to_r1_r2(
            r1=evecs_davidson[:18, root], 
            r2=evecs_davidson[18:, root],
            nocc=inputs.nocc,
            nvirt=inputs.nvirt
        )
        expanded_r1 = r1#dh.expand_r1(r1, nocc=6, nvirt=6)
        expanded_r2 = r2#dh.expand_r2(r2, nocc=6, nvirt=6)


        W = inputs.tei
        D2R2_eff = drive_R2_projection(W,inputs.o,inputs.v,expanded_r1,expanded_r2,inputs.t2amps,D3)
        D2T2_capped_E = 0.25*np.einsum('jiab,abji',D2R2_eff, expanded_r2.transpose(2,3,0,1))
        print("Root num:", root, "capped E (eV):",(D2T2_capped_E+omega)*27.2114)
        print("\n")
        sqrbrkT_values.append((D2T2_capped_E+omega)*27.2114)

    sqrbrkT_values = np.asarray(
        np.real_if_close(sqrbrkT_values),
        dtype=float,
    )
    print("sqrbrak t values in eV: ", sqrbrkT_values)
    known_t_corrected_ev = np.array([
        1.292629319,
        1.292629366,
        3.575264684,
        3.575264744,
        5.745657799,
        7.673614785,
        7.682439325,
        8.949542946,
        9.014173484,
        11.31320,
        11.31319783,
        11.76289809,
        14.04722992,
        14.04900237,
        14.51504931,
    ])

    tolerance_ev = 1.0e-3


    assert np.allclose(
        sqrbrkT_values,
        known_t_corrected_ev,
        atol=tolerance_ev,
        rtol=0.0,
    ), (
        "Davidson excitation energies do not match the known solutions.\n"
    )
