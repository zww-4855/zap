import numpy as np
import scipy.linalg as la


def modified_gram_schmidt(X, Q=None, tol=1e-12, reorthogonalize=True):
    """
    Orthonormalize columns of X against existing basis Q and against each other.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Candidate vectors to orthonormalize.
    Q : ndarray, shape (n, k), optional
        Existing orthonormal basis to orthogonalize against.
    tol : float
        Drop vectors with norm below this threshold.
    reorthogonalize : bool
        Whether to do a second Gram-Schmidt pass.

    Returns
    -------
    Vnew : ndarray, shape (n, p)
        Orthonormalized accepted vectors.
    """
    X = np.array(X, dtype=complex, copy=True)

    if X.ndim == 1:
        X = X[:, None]

    n, m = X.shape
    accepted = []

    for j in range(m):
        v = X[:, j].copy()

        # Orthogonalize against existing basis Q
        if Q is not None and Q.size > 0:
            v -= Q @ (Q.conj().T @ v)

        # Orthogonalize against newly accepted vectors
        for q in accepted:
            v -= q * np.vdot(q, v)

        if reorthogonalize:
            if Q is not None and Q.size > 0:
                v -= Q @ (Q.conj().T @ v)

            for q in accepted:
                v -= q * np.vdot(q, v)

        norm = la.norm(v)

        if norm > tol:
            accepted.append(v / norm)

    if len(accepted) == 0:
        return np.empty((n, 0), dtype=complex)

    return np.column_stack(accepted)


def davidson(
    A,
    n_roots=1,
    max_iter=100,
    max_subspace=40,
    tol=1e-10,
    initial_guess=None,
    diag=None,
    verbose=True,
):
    """
    Block Davidson diagonalization for the lowest eigenvalues of a Hermitian matrix.

    Parameters
    ----------
    A : ndarray or callable
        Matrix A, or a function matvec(x) returning A @ x.
    n_roots : int
        Number of lowest eigenpairs to compute.
    max_iter : int
        Maximum Davidson iterations.
    max_subspace : int
        Maximum number of subspace vectors before restart.
    tol : float
        Residual norm convergence threshold.
    initial_guess : ndarray, optional
        Initial subspace guess vectors, shape (n, m).
    diag : ndarray, optional
        Diagonal of A. Used in Davidson preconditioning.
    verbose : bool
        Print iteration information.

    Returns
    -------
    evals : ndarray, shape (n_roots,)
        Approximate lowest eigenvalues.
    evecs : ndarray, shape (n, n_roots)
        Approximate eigenvectors.
    residual_norms : ndarray, shape (n_roots,)
        Final residual norms.
    """

    # ------------------------------------------------------------
    # Define matrix-vector product
    # ------------------------------------------------------------
    if callable(A):
        matvec = A
        if diag is None:
            raise ValueError("If A is callable, you must provide diag.")
        n = len(diag)
    else:
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]
        matvec = lambda x: A @ x
        if diag is None:
            diag = np.diag(A).real

    diag = np.asarray(diag, dtype=float)

    if n_roots > n:
        raise ValueError("n_roots cannot exceed matrix dimension.")

    # ------------------------------------------------------------
    # Initial guess subspace
    # ------------------------------------------------------------
    if initial_guess is None:
        # Use unit vectors corresponding to the smallest diagonal elements
        idx = np.argsort(diag)[: max(n_roots, 2)]
        initial_guess = np.eye(n, dtype=complex)[:, idx]

    V = modified_gram_schmidt(initial_guess, tol=1e-14)

    if V.shape[1] < n_roots:
        raise ValueError("Initial guess produced too few independent vectors.")

    # Store A|V> to avoid recomputing products
    AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])

    evals = None
    evecs = None
    residual_norms = None

    for it in range(1, max_iter + 1):

        # ------------------------------------------------------------
        # Step 2: projected subspace Hamiltonian
        #
        # Hbar = V^dagger A V
        # ------------------------------------------------------------
        Hbar = V.conj().T @ AV

        # Symmetrize to remove numerical noise
        Hbar = 0.5 * (Hbar + Hbar.conj().T)

        # ------------------------------------------------------------
        # Step 3: solve subspace eigenproblem
        #
        # Hbar C = C Omega
        # ------------------------------------------------------------
        theta, C = la.eigh(Hbar)

        theta = theta[:n_roots]
        C = C[:, :n_roots]

        # ------------------------------------------------------------
        # Step 4: update Ritz vectors
        #
        # psi_k = V C_k
        # ------------------------------------------------------------
        ritz_vectors = V @ C

        # A psi_k = A V C_k
        A_ritz_vectors = AV @ C

        # ------------------------------------------------------------
        # Step 5: residuals
        #
        # r_k = A psi_k - theta_k psi_k
        # ------------------------------------------------------------
        residuals = A_ritz_vectors - ritz_vectors * theta[None, :]
        residual_norms = np.array([la.norm(residuals[:, k]) for k in range(n_roots)])

        if verbose:
            eig_str = " ".join(f"{x: .12f}" for x in theta.real)
            res_str = " ".join(f"{r: .3e}" for r in residual_norms)
            print(
                f"Iter {it:3d} | subspace dim = {V.shape[1]:3d} | "
                f"evals = {eig_str} | residuals = {res_str}"
            )

        # Check convergence
        if np.all(residual_norms < tol):
            evals = theta.real
            evecs = ritz_vectors
            break

        # ------------------------------------------------------------
        # Step 6: Davidson preconditioning
        #
        # Approximate solve:
        #
        #     (A - theta_k I) t_k = -r_k
        #
        # using only the diagonal of A:
        #
        #     t_k ~= r_k / (theta_k - A_diag)
        # ------------------------------------------------------------
        correction_vectors = []

        for k in range(n_roots):
            if residual_norms[k] < tol:
                continue

            denom = theta[k].real - diag

            # Avoid division by small denominators
            small = np.abs(denom) < 1e-12
            denom[small] = np.sign(denom[small] + 1e-16) * 1e-12

            t = residuals[:, k] / denom

            correction_vectors.append(t)

        if len(correction_vectors) == 0:
            evals = theta.real
            evecs = ritz_vectors
            break

        T = np.column_stack(correction_vectors)

        # Orthogonalize new correction vectors against existing subspace
        T = modified_gram_schmidt(T, Q=V, tol=1e-12)

        # If all correction vectors were linearly dependent, restart
        if T.shape[1] == 0:
            if verbose:
                print("No independent correction vectors found. Restarting.")

            V = modified_gram_schmidt(ritz_vectors, tol=1e-14)
            AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])
            continue

        # Expand subspace
        V = np.column_stack([V, T])
        AT = np.column_stack([matvec(T[:, j]) for j in range(T.shape[1])])
        AV = np.column_stack([AV, AT])

        # ------------------------------------------------------------
        # Restart if subspace becomes too large
        # ------------------------------------------------------------
        if V.shape[1] > max_subspace:
            if verbose:
                print("Restarting subspace.")

            V = modified_gram_schmidt(ritz_vectors, tol=1e-14)
            AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])

    else:
        evals = theta.real
        evecs = ritz_vectors
        if verbose:
            print("Davidson did not fully converge within max_iter.")

    # Normalize final eigenvectors
    evecs = modified_gram_schmidt(evecs, tol=1e-14)

    return evals, evecs, residual_norms


if __name__ == "__main__":

    np.random.seed(7)

    # ------------------------------------------------------------
    # Example: construct a Hermitian matrix with known structure
    # ------------------------------------------------------------
    #n = 200

    #diagonal = np.linspace(0.0, 50.0, n)
    #A = np.diag(diagonal)

    # Add a small dense Hermitian perturbation
    #X = np.random.randn(n, n)
    #perturbation = 0.05 * (X + X.T)

    #A = A + perturbation

    # Load matrix representing CH+ molecule
    re_path = "/Users/zwu/Desktop/work/iterative_EOM/CH_Hbars/re_full.txt"
    import read_info as ri
    Hbar = ri.read_Hbar_matrix_from_txt(re_path, dim=117, dtype=float)
    # ------------------------------------------------------------
    # Davidson diagonalization
    # ------------------------------------------------------------
    evals_davidson, evecs_davidson, res = davidson(
        Hbar,
        n_roots=4,
        max_iter=100,
        max_subspace=30,
        tol=1e-10,
        diag=np.diag(Hbar),
        verbose=True,
    )

    print("\nDavidson eigenvalues:")
    print(evals_davidson)

    # ------------------------------------------------------------
    # Exact diagonalization for comparison
    # ------------------------------------------------------------
    evals_exact, evecs_exact = la.eigh(Hbar)

    print("\nExact lowest eigenvalues:")
    print(evals_exact[:4])
    print("\nExact eigenvectors:")
    print(evecs_exact[:, :1])
    print("\nDavidson eigenvectors:")
    print(evecs_davidson[18:, :1])
    #print("\nAbsolute errors:")
    #print(np.abs(evals_davidson - evals_exact[:4]))
    #sys.exit()
    
    
    import run_test as rt
    nocc,nvirt,tei,t2amps,t1amps,D3,o,v,mo_energies = rt.helper()
    print('size of t2: ',np.shape(t2amps))


    # I need to be able to expand the R1 and R2 pieces of the Davidson eigenvector into the full tensors. T
    # The R1 piece is straightforward, but the R2 piece is more complicated due to spin adaptation and unique excitation ordering.
    # This block of code will also test the opposite operation: reducing the expanded R1 and R2 tensors back to the original Davidson eigenvector pieces.
    import dash_helper as dh
    print("r1:",evecs_davidson[:18, 0])
    r1 = dh.expand_r1(evecs_davidson[:18, 0],nocc, nvirt)
    root = 0
    r1, r2 = dh.parse_davidson_eigenvector_to_r1_r2(
        r1=evecs_davidson[:18, root], 
        r2=evecs_davidson[18:, root],
        nocc=nocc,
        nvirt=nvirt
    )

    print("Testing expanded -> reduced r1")
    small_r1_original = evecs_davidson[:18, root]

    expanded_r1 = dh.expand_r1(small_r1_original, nocc=6, nvirt=6)

    small_r1_recovered = dh.reduce_r1(expanded_r1, nocc=6, nvirt=6)

    print(np.allclose(small_r1_original, small_r1_recovered))

    print("Testing expanded -> reduced r2")
    small_r2_original = evecs_davidson[18:, root]

    expanded_r2 = dh.expand_r2(small_r2_original, nocc=6, nvirt=6)

    small_r2_recovered = dh.reduce_r2(expanded_r2, nocc=6, nvirt=6)

    print("Testing expanded -> reduced r2:")
    print(np.allclose(small_r2_original, small_r2_recovered))
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Now, test this by performing the [T-6] energy correction before I do the projection of R3 back into R2/R1
    import build_R3 as br3
    import copy
    import projection_R3 as pr3


    W = tei
    #D3C3_WC1T2, D3C3_WC2 = br3.build_eom_sqrbrakT_resid(W,t2amps,expanded_r1,expanded_r2,o,v)
    
    #D3C3 = D3C3_WC1T2 + D3C3_WC2
    #C3 = copy.deepcopy(D3C3) * D3 
    # all tamps, C vectors are in (o,o,v,v) order
    #tmp=D3C3_WC2*D3D3C3_WC2stat = (1/36.0)*np.einsum("ijkabc,abcijk->",D3C3_WC2,tmp.transpose(3,4,5,0,1,2),optimize="optimal")
    #
    D2R2_eff = pr3.drive_R2_projection(W,o,v,expanded_r1,expanded_r2,t2amps,D3)
    D2T2_capped_E = 0.25*np.einsum('jiab,abji',D2R2_eff, expanded_r2.transpose(2,3,0,1))

    print("EOM [T-6] energy correction D2T2_capped_E:",D2T2_capped_E)

    # this function calls all of R3-> R1 functionality
    D1R1_eff =  pr3.drive_R1_projection(W,o,v,expanded_r1,expanded_r2,t2amps,D3)
    top_dias = np.einsum("ia,jb->",D1R1_eff,expanded_r1.transpose(1,0),optimize="optimal")
    print("EOM [T-6] energy correction from R3->R1 projection:",top_dias)


