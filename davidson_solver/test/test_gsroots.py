import pytest
#from davidson_solver.davidson import davidson
from ..davidson import davidson
import numpy as np

def test_EOMUCCSD_spectra(): 

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
        max_subspace=60,
        tol=1e-12,
        diag=np.diag(Hbar),
        verbose=True,
    )

    print("\nDavidson eigenvalues:")
    print(evals_davidson)
    print('resids:',res)
    threshold = 1.0e-8

    res = np.asarray(res, dtype=float)

    print("resids:", res)

    assert np.all(res < threshold), (
        f"Some Davidson residuals exceed {threshold:.1e}.\n"
        f"Maximum residual: {np.max(res):.6e}\n"
        f"Failed roots: {np.where(res >= threshold)[0].tolist()}\n"
        f"Failed residuals: {res[res >= threshold]}"
    )

    known_solutions_ev = np.array([
        1.29297,
        1.29297,
        3.57543,
        3.57543,
        5.75311,
        7.69884,
        7.69884,
        8.95610,
        9.07654,
        11.32050,
        11.32050,
        11.76620,
        14.14440,
        14.14440,
        14.51710,
    ])

    davidson_solutions_ev = (
        np.asarray(evals_davidson[:15], dtype=float) * 27.2114
    )

    tolerance_ev = 1.0e-4

    # Verify that Davidson EOM UCCSD equals what eigenvalues in eV that I have in spreadsheet
    assert np.allclose(
        davidson_solutions_ev,
        known_solutions_ev,
        atol=tolerance_ev,
        rtol=0.0,
    ), (
        "Davidson excitation energies do not match the known solutions.\n"
        f"Davidson (eV):\n{davidson_solutions_ev}\n"
        f"Known (eV):\n{known_solutions_ev}\n"
        f"Absolute errors (eV):\n"
        f"{np.abs(davidson_solutions_ev - known_solutions_ev)}\n"
        f"Failed roots:\n"
        f"{np.where(np.abs(davidson_solutions_ev - known_solutions_ev) > tolerance_ev)[0]}"
    )



    #assert True

def test_add_positive_numbers():
    # Simple assertion
    assert 2+5 == 5
