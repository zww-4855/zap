import numpy as np

def expand_r1(small_r1, nocc=6, nvirt=6):
    """
    Print R1-like excitation labels of the form

        a^ i    |

    using spin-orbital indices.

    occupied: 0, 1, ..., nocc - 1
    virtual:  nocc, ..., nocc + nvirt - 1

    Spin conservation is enforced by parity:
        even occupied -> even virtual
        odd occupied  -> odd virtual
    """

    expanded_r1 = np.zeros((nvirt, nocc))
    counter = 0
    occ = list(range(nocc))
    virt = list(range(nocc, nocc + nvirt))

    for i in occ:
        for a in virt:

            # Enforce spin conservation by parity
            if a % 2 != i % 2:
                continue

            amp_value = small_r1[counter]
            A = a - nocc
            expanded_r1[A, i] = amp_value

            counter += 1
            
            print(f"{a}^ {i}\t|")

    return expanded_r1.transpose(1,0)


def expand_r2(small_r2,nocc=6, nvirt=6):
    """
    Print R2-like excitation labels of the form

        a^ i; b^ j    |

    using spin-orbital indices.

    occupied: 0, 1, ..., nocc - 1
    virtual:  nocc, ..., nocc + nvirt - 1

    Spin conservation is enforced by parity:
        even occupied -> even virtual
        odd occupied  -> odd virtual

    For same-spin occupied pairs, only a < b is printed to avoid duplicates.
    """

    expanded_r2 = np.zeros((nvirt, nocc, nvirt,  nocc))
    counter = 0
    occ = list(range(nocc))
    virt = list(range(nocc, nocc + nvirt))

    for i in occ:
        for j in occ:

            # Avoid duplicate occupied pairs
            if j <= i:
                continue

            for a in virt:
                for b in virt:

                    # Enforce spin conservation by parity
                    if a % 2 != i % 2:
                        continue

                    if b % 2 != j % 2:
                        continue

                    # For same-spin pairs, avoid duplicate virtual pairs
                    # Example: keep 6^ 0; 8^ 2, but skip 8^ 0; 6^ 2
                    if i % 2 == j % 2:
                        if b <= a:
                            continue
                    amp_value = small_r2[counter]
                    A = a - nocc
                    B = b - nocc
                    expanded_r2[A, i, B, j] = amp_value
                    expanded_r2[A, j, B, i] = -1.0 * amp_value
                    expanded_r2[B, i, A, j] = -1.0 * amp_value
                    expanded_r2[B, j, A, i] = amp_value

                    counter += 1
                    
                    print(f"{a}^ {i}; {b}^ {j}\t|")

    return expanded_r2.transpose(1,3,0,2)

def parse_davidson_eigenvector_to_r1_r2(r1, r2, nocc=6, nvirt=6):
    """
    Parse the Davidson eigenvector pieces into expanded R1 and R2 tensors.

    Parameters
    ----------
    r1 : ndarray
        The R1 part of the Davidson eigenvector.
        For nocc=6, nvirt=6, this should have length 18.

    r2 : ndarray
        The R2 part of the Davidson eigenvector.
        For nocc=6, nvirt=6, this should have length 99 using your spin-adapted
        / unique excitation ordering.

    nocc : int
        Number of occupied spin orbitals.

    nvirt : int
        Number of virtual spin orbitals.

    Returns
    -------
    expanded_r1 : ndarray
        Expanded R1 tensor with shape (nocc, nvirt).

    expanded_r2 : ndarray
        Expanded R2 tensor with shape (nocc, nocc, nvirt, nvirt).
    """


    expanded_r1 = expand_r1(r1, nocc=nocc, nvirt=nvirt)
    expanded_r2 = expand_r2(r2, nocc=nocc, nvirt=nvirt)

    return expanded_r1, expanded_r2


    import numpy as np


def reduce_r1(expanded_r1, nocc=6, nvirt=6, print_labels=False):
    """
    Undo the operation performed by expand_r1.

    Takes an expanded R1 tensor with shape

        expanded_r1[i, a_local]

    and returns the reduced spin-conserving R1 vector in the same order used by
    expand_r1.

    Parameters
    ----------
    expanded_r1 : ndarray
        Expanded R1 tensor with shape (nocc, nvirt).

    nocc : int
        Number of occupied spin orbitals.

    nvirt : int
        Number of virtual spin orbitals.

    print_labels : bool
        If True, print the excitation labels in the same order as expand_r1.

    Returns
    -------
    small_r1 : ndarray
        Reduced R1 vector.
    """

    expanded_r1 = np.asarray(expanded_r1)

    expected_shape = (nocc, nvirt)
    if expanded_r1.shape != expected_shape:
        raise ValueError(
            f"expanded_r1 has shape {expanded_r1.shape}, "
            f"but expected {expected_shape}."
        )

    small_r1 = []

    occ = list(range(nocc))
    virt = list(range(nocc, nocc + nvirt))

    for i in occ:
        for a in virt:

            # Enforce same spin-conservation rule as expand_r1
            if a % 2 != i % 2:
                continue

            A = a - nocc

            amp_value = expanded_r1[i, A]
            small_r1.append(amp_value)

            if print_labels:
                print(f"{a}^ {i}\t|")

    return np.array(small_r1)

import numpy as np


def reduce_r2(expanded_r2, nocc=6, nvirt=6, print_labels=False, check_antisymmetry=False):
    """
    Undo the operation performed by expand_r2.

    Takes an expanded R2 tensor with shape

        expanded_r2[i, j, A, B]

    where A = a - nocc and B = b - nocc, and returns the reduced
    spin-conserving R2 vector in the same ordering used by expand_r2.

    Parameters
    ----------
    expanded_r2 : ndarray
        Expanded R2 tensor with shape (nocc, nocc, nvirt, nvirt).

    nocc : int
        Number of occupied spin orbitals.

    nvirt : int
        Number of virtual spin orbitals.

    print_labels : bool
        If True, print the excitation labels in the same order as expand_r2.

    check_antisymmetry : bool
        If True, check that expanded_r2 satisfies the expected antisymmetry:
            R_ij^ab = -R_ji^ab = -R_ij^ba = R_ji^ba

    Returns
    -------
    small_r2 : ndarray
        Reduced R2 vector in the same ordering expected by expand_r2.
    """

    expanded_r2 = np.asarray(expanded_r2)

    expected_shape = (nocc, nocc, nvirt, nvirt)
    if expanded_r2.shape != expected_shape:
        raise ValueError(
            f"expanded_r2 has shape {expanded_r2.shape}, "
            f"but expected {expected_shape}."
        )

    if check_antisymmetry:
        err_ij = np.max(np.abs(expanded_r2 + expanded_r2.swapaxes(0, 1)))
        err_ab = np.max(np.abs(expanded_r2 + expanded_r2.swapaxes(2, 3)))
        err_ijab = np.max(
            np.abs(expanded_r2 - expanded_r2.swapaxes(0, 1).swapaxes(2, 3))
        )

        print(f"max |R_ijab + R_jiab| = {err_ij:.3e}")
        print(f"max |R_ijab + R_ijba| = {err_ab:.3e}")
        print(f"max |R_ijab - R_jiba| = {err_ijab:.3e}")

    small_r2 = []

    occ = list(range(nocc))
    virt = list(range(nocc, nocc + nvirt))

    for i in occ:
        for j in occ:

            # Same occupied-pair rule as expand_r2
            if j <= i:
                continue

            for a in virt:
                for b in virt:

                    # Same spin-conservation rules as expand_r2
                    if a % 2 != i % 2:
                        continue

                    if b % 2 != j % 2:
                        continue

                    # Same same-spin duplicate-removal rule as expand_r2
                    if i % 2 == j % 2:
                        if b <= a:
                            continue

                    A = a - nocc
                    B = b - nocc

                    amp_value = expanded_r2[i, j, A, B]
                    small_r2.append(amp_value)

                    if print_labels:
                        print(f"{a}^ {i}; {b}^ {j}\t|")

    return np.array(small_r2, dtype=expanded_r2.dtype)


def antisym_T2(Roovv, nocc, nvir):
    # antisymmetrize the residual
    Roovv_anti = np.zeros((nocc, nocc, nvir, nvir))
    Roovv_anti += np.einsum("ijab->ijab", Roovv)
    Roovv_anti -= np.einsum("ijab->jiab", Roovv)
    Roovv_anti -= np.einsum("ijab->ijba", Roovv)
    Roovv_anti += np.einsum("ijab->jiba", Roovv)
    return Roovv_anti