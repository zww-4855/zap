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
    assert True

def test_add_positive_numbers():
    # Simple assertion
    assert 2+3 == 5
