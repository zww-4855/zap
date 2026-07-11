def print_r2_labels(nocc=6, nvirt=6, include_bar=True):
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

                    if include_bar:
                        print(f"{a}^ {i}; {b}^ {j}\t|")
                    else:
                        print(f"{a}^ {i}; {b}^ {j}")


print_r2_labels(6,6,False)
