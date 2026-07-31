import warnings

import numpy as np
import scipy.linalg as la


def davidson(
    A,
    n_roots=1,
    max_iter=200,
    max_subspace=40,
    tol=1.0e-10,
    initial_guess=None,
    diag=None,
    verbose=True,
    denominator_tolerance=1.0e-12,
    orthogonalization_tolerance=1.0e-13,
    raise_on_nonconvergence=True,
):
    """
    Block Davidson diagonalization for the lowest eigenvalues of a
    Hermitian matrix.

    Convergence requires the residual norm of every requested root to satisfy

        ||A x_k - theta_k x_k||_2 <= tol.

    Parameters
    ----------
    A : ndarray or callable
        Hermitian matrix A, or a callable implementing A @ x.

    n_roots : int
        Number of lowest eigenpairs to compute.

    max_iter : int
        Maximum number of Davidson iterations.

    max_subspace : int
        Maximum subspace dimension before a thick restart.

        For block Davidson, this should generally be at least 2*n_roots
        and preferably about 3*n_roots or larger.

    tol : float
        Absolute residual norm convergence threshold.

    initial_guess : ndarray, optional
        Initial guess vectors with shape (n, n_guess), or one vector
        with shape (n,).

    diag : ndarray, optional
        Diagonal approximation to A used in the Davidson preconditioner.
        Required when A is callable.

    verbose : bool
        Print iteration information.

    denominator_tolerance : float
        Minimum magnitude allowed in the diagonal preconditioner.

    orthogonalization_tolerance : float
        Threshold used when removing linearly dependent correction vectors.

    raise_on_nonconvergence : bool
        Raise RuntimeError if every requested root does not converge.

    Returns
    -------
    evals : ndarray, shape (n_roots,)
        Converged Ritz eigenvalues.

    evecs : ndarray, shape (n, n_roots)
        Corresponding Ritz eigenvectors.

    residual_norms : ndarray, shape (n_roots,)
        Residual norm for every returned eigenpair.
    """

    # ------------------------------------------------------------
    # Define the matrix-vector operation and diagonal
    # ------------------------------------------------------------
    if callable(A):
        if diag is None:
            raise ValueError(
                "diag must be supplied when A is provided as a callable."
            )

        diag = np.asarray(diag)
        if diag.ndim != 1:
            raise ValueError(
                f"diag must be one-dimensional; got shape {diag.shape}."
            )

        n = diag.size
        matvec = A

    else:
        A = np.asarray(A)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"A must be a square matrix; got shape {A.shape}."
            )

        n = A.shape[0]

        if not np.allclose(A, A.conj().T, atol=1.0e-12, rtol=1.0e-10):
            max_error = np.max(np.abs(A - A.conj().T))
            raise ValueError(
                "This Davidson implementation requires a Hermitian matrix. "
                f"Maximum Hermiticity error: {max_error:.3e}."
            )

        matvec = lambda x: A @ x

        if diag is None:
            diag = np.diag(A)

    diag = np.asarray(np.real_if_close(diag), dtype=float)

    if not isinstance(n_roots, int) or n_roots < 1:
        raise ValueError("n_roots must be a positive integer.")

    if n_roots > n:
        raise ValueError(
            f"n_roots={n_roots} exceeds the matrix dimension n={n}."
        )

    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    # ------------------------------------------------------------
    # Ensure the subspace is large enough for block Davidson
    # ------------------------------------------------------------
    recommended_subspace = min(n, max(3 * n_roots, n_roots + 4))

    if max_subspace < 2 * n_roots and max_subspace < n:
        old_max_subspace = max_subspace
        max_subspace = recommended_subspace

        warnings.warn(
            f"max_subspace={old_max_subspace} is too small for "
            f"n_roots={n_roots}. Increasing max_subspace to "
            f"{max_subspace}.",
            RuntimeWarning,
        )

    max_subspace = min(max_subspace, n)

    if max_subspace < n_roots:
        raise ValueError(
            f"max_subspace={max_subspace} cannot be smaller than "
            f"n_roots={n_roots}."
        )

    # ------------------------------------------------------------
    # Initial subspace
    # ------------------------------------------------------------
    if initial_guess is None:
        guess_indices = np.argsort(diag)[:n_roots]
        initial_guess = np.eye(n, dtype=complex)[:, guess_indices]
    else:
        initial_guess = np.asarray(initial_guess, dtype=complex)

        if initial_guess.ndim == 1:
            initial_guess = initial_guess[:, None]

        if initial_guess.shape[0] != n:
            raise ValueError(
                f"Initial guesses have dimension {initial_guess.shape[0]}, "
                f"but the matrix dimension is {n}."
            )

    V = modified_gram_schmidt(
        initial_guess,
        tol=orthogonalization_tolerance,
    )

    if V.shape[1] < n_roots:
        raise ValueError(
            "The initial guess contains fewer than n_roots independent "
            f"vectors: found {V.shape[1]}, need {n_roots}."
        )

    def apply_to_block(X):
        """Apply A independently to every column of X."""

        return np.column_stack(
            [np.asarray(matvec(X[:, col])) for col in range(X.shape[1])]
        )

    AV = apply_to_block(V)

    evals = None
    evecs = None
    residual_norms = None
    converged = False

    # ------------------------------------------------------------
    # Davidson iterations
    # ------------------------------------------------------------
    for iteration in range(1, max_iter + 1):

        # Project A into the current subspace.
        projected_matrix = V.conj().T @ AV

        # Remove small numerical violations of Hermiticity.
        projected_matrix = 0.5 * (
            projected_matrix + projected_matrix.conj().T
        )

        # Solve the complete projected problem.
        theta_all, coefficients_all = la.eigh(projected_matrix)

        theta = theta_all[:n_roots]
        coefficients = coefficients_all[:, :n_roots]

        # Ritz vectors and their matrix-vector products.
        ritz_vectors = V @ coefficients
        A_ritz_vectors = AV @ coefficients

        # Residuals for every requested root.
        residuals = (
            A_ritz_vectors
            - ritz_vectors * theta[np.newaxis, :]
        )

        residual_norms = la.norm(residuals, axis=0)
        root_converged = residual_norms <= tol

        if verbose:
            print(
                f"\nIteration {iteration:3d} | "
                f"subspace dimension = {V.shape[1]:3d}"
            )
            print(" root          eigenvalue              residual norm")

            for root in range(n_roots):
                status = "converged" if root_converged[root] else ""
                print(
                    f" {root:4d}  "
                    f"{theta[root].real: 20.12f}  "
                    f"{residual_norms[root]: 14.6e}  "
                    f"{status}"
                )

        # Require every requested root to satisfy the tolerance.
        if np.all(root_converged):
            evals = theta.real.copy()
            evecs = ritz_vectors.copy()
            converged = True
            break

        # ------------------------------------------------------------
        # Build correction vectors only for unconverged roots
        # ------------------------------------------------------------
        correction_vectors = []

        for root in range(n_roots):
            if root_converged[root]:
                continue

            denominator = theta[root].real - diag

            denominator_sign = np.where(
                denominator >= 0.0,
                1.0,
                -1.0,
            )

            safe_denominator = np.where(
                np.abs(denominator) < denominator_tolerance,
                denominator_sign * denominator_tolerance,
                denominator,
            )

            correction = residuals[:, root] / safe_denominator
            correction_vectors.append(correction)

        if not correction_vectors:
            raise RuntimeError(
                "No correction vectors were generated even though not all "
                "roots satisfy the convergence threshold."
            )

        T = np.column_stack(correction_vectors)

        # Orthogonalize the corrections against V and against each other.
        T = modified_gram_schmidt(
            T,
            Q=V,
            tol=orthogonalization_tolerance,
            reorthogonalize=True,
        )

        if T.shape[1] == 0:
            # Restart with the current Ritz vectors, then attempt to generate
            # independent random directions.
            if verbose:
                print(
                    "All correction vectors were linearly dependent. "
                    "Restarting with the current Ritz vectors."
                )

            V = modified_gram_schmidt(
                ritz_vectors,
                tol=orthogonalization_tolerance,
                reorthogonalize=True,
            )
            AV = apply_to_block(V)

            # Add independent random directions if possible.
            n_extra = min(n_roots, n - V.shape[1])

            if n_extra > 0:
                rng = np.random.default_rng(12345 + iteration)
                random_vectors = (
                    rng.standard_normal((n, n_extra))
                    + 1j * rng.standard_normal((n, n_extra))
                )

                random_vectors = modified_gram_schmidt(
                    random_vectors,
                    Q=V,
                    tol=orthogonalization_tolerance,
                    reorthogonalize=True,
                )

                if random_vectors.shape[1] > 0:
                    V = np.column_stack((V, random_vectors))
                    AV = np.column_stack(
                        (AV, apply_to_block(random_vectors))
                    )

            continue

        # ------------------------------------------------------------
        # Expand or thick-restart the subspace
        # ------------------------------------------------------------
        if V.shape[1] + T.shape[1] <= max_subspace:
            # Normal expansion.
            V = np.column_stack((V, T))
            AV = np.column_stack((AV, apply_to_block(T)))

        else:
            # Thick restart.
            #
            # Crucially, preserve both:
            #   1. the current Ritz vectors, and
            #   2. the newly generated correction vectors.
            #
            # The original code discarded T immediately when the subspace
            # limit was exceeded.
            restart_vectors = np.column_stack((ritz_vectors, T))

            V = modified_gram_schmidt(
                restart_vectors,
                tol=orthogonalization_tolerance,
                reorthogonalize=True,
            )

            if V.shape[1] > max_subspace:
                V = V[:, :max_subspace]

            AV = apply_to_block(V)

            if verbose:
                print(
                    "Thick restart performed; retained "
                    f"{V.shape[1]} Ritz/correction vectors."
                )

    # ------------------------------------------------------------
    # Handle nonconvergence
    # ------------------------------------------------------------
    if not converged:
        evals = theta.real.copy()
        evecs = ritz_vectors.copy()

    # ------------------------------------------------------------
    # Recompute residuals for the actual returned eigenvectors
    # ------------------------------------------------------------
    #
    # Do not run Gram-Schmidt on evecs here. Ritz vectors produced from an
    # orthonormal V and the Hermitian projected eigensystem are already
    # orthonormal. Mixing them after convergence would invalidate their
    # eigenvalue pairing and previously calculated residuals.
    #
    A_evecs = apply_to_block(evecs)

    final_residuals = (
        A_evecs
        - evecs * evals[np.newaxis, :]
    )

    residual_norms = la.norm(final_residuals, axis=0)
    final_converged = residual_norms <= tol

    if verbose:
        print("\nFinal Davidson residuals")
        print(" root          eigenvalue              residual norm")

        for root in range(n_roots):
            status = "converged" if final_converged[root] else "NOT CONVERGED"
            print(
                f" {root:4d}  "
                f"{evals[root]: 20.12f}  "
                f"{residual_norms[root]: 14.6e}  "
                f"{status}"
            )

    if not np.all(final_converged):
        failed_roots = np.where(~final_converged)[0]

        message = (
            "Davidson did not converge every requested root to the "
            f"specified tolerance {tol:.3e}. "
            f"Unconverged roots: {failed_roots.tolist()}. "
            f"Residuals: {residual_norms[failed_roots]}."
        )

        if raise_on_nonconvergence:
            raise RuntimeError(message)

        if verbose:
            print(message)

    return evals, evecs, residual_norms
