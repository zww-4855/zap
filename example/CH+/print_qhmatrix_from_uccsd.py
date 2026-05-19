"""Print CH+ QSC-EOM Hamiltonian matrices from saved UCCSD amplitudes."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import casci
import qsceom
import qsceom_exact


HARTREE_RE_BOHR = 2.13713
HMATRIX_DIR = PROJECT_ROOT / "Hmatrix_CH+"


class _MatrixCaptured(Exception):
    pass


def _as_array(coords):
    return np.array(coords, dtype=float)


def _problem(distance_factor):
    return {
        "symbols": ["C", "H"],
        "geometry": _as_array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, float(distance_factor) * HARTREE_RE_BOHR],
            ]
        ),
        "active_electrons": 6,
        "active_orbitals": 6,
        "charge": 1,
        "basis": "6-31g",
        "unit": "bohr",
    }


def _cases():
    return [
        {
            "name": "CH+re",
            "distance_factor": 1.0,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+re" / "CH+ Trotterized" / "t1_t2.txt",
            "module": qsceom,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+re).txt",
            "exact": False,
        },
        {
            "name": "CH+1.5re",
            "distance_factor": 1.5,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+1.5re" / "CH+ Trotterized" / "t1_t2.txt",
            "module": qsceom,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+1.5re).txt",
            "exact": False,
        },
        {
            "name": "CH+2re",
            "distance_factor": 2.0,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+2re" / "CH+ Trotterized" / "t1_t2.txt",
            "module": qsceom,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+2re).txt",
            "exact": False,
        },
        {
            "name": "CH+re_full_operator",
            "distance_factor": 1.0,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+re" / "CH+ Full operator" / "t1_t2.txt",
            "module": qsceom_exact,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+re_full_operator).txt",
            "exact": True,
        },
        {
            "name": "CH+1.5re_full_operator",
            "distance_factor": 1.5,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+1.5re" / "CH+ Full operator" / "t1_t2.txt",
            "module": qsceom_exact,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+1.5re_full_operator).txt",
            "exact": True,
        },
        {
            "name": "CH+2re_full_operator",
            "distance_factor": 2.0,
            "amp_file": PROJECT_ROOT / "outputs" / "CH+2re" / "CH+ Full operator" / "t1_t2.txt",
            "module": qsceom_exact,
            "output_file": HMATRIX_DIR / "qHMatrix(CH+2re_full_operator).txt",
            "exact": True,
        },
    ]


def _canon_pair(p, q):
    p = int(p)
    q = int(q)
    sign = 1.0
    if (p % 2) != (q % 2):
        if p % 2 == 1:
            p, q = q, p
            sign *= -1.0
    else:
        if p > q:
            p, q = q, p
            sign *= -1.0
    return p, q, sign


def _load_params_from_t1_t2(amp_file, cfg, qubits):
    singles, doubles = qml.qchem.excitations(cfg["active_electrons"], qubits)
    expected = len(doubles) + len(singles)

    amp_values = []
    for raw in amp_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        left, right = line.split("|", 1)
        if "^" not in left:
            continue
        try:
            amp_values.append(float(right.strip()))
        except ValueError:
            continue

    if len(amp_values) < expected:
        raise ValueError(
            f"Not enough amplitudes in {amp_file}. Found {len(amp_values)}, expected {expected}."
        )
    amp_values = amp_values[:expected]

    t2 = np.zeros(len(doubles), dtype=float)
    for idx, (i, j, a, b) in enumerate(doubles):
        _, _, s_ab = _canon_pair(a, b)
        _, _, s_ij = _canon_pair(i, j)
        t2[idx] = amp_values[idx] / (s_ab * s_ij)

    t1_start = len(doubles)
    t1 = np.asarray(amp_values[t1_start : t1_start + len(singles)], dtype=float)
    return np.concatenate([t1, t2])


def _format_matrix_value(value):
    cval = complex(value)
    if abs(cval.imag) <= 1e-12:
        return f"{cval.real:.16e}"
    return f"{cval.real:.16e}{cval.imag:+.16e}j"


def _write_qhmatrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix)
    lines = ["\t".join(_format_matrix_value(value) for value in row) for row in arr]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE_QHMATRIX={path}")


def _capture_trotterized_matrix(case, cfg, hamiltonian, qubits, params):
    module = case["module"]
    module._require_quantum_deps()
    original_eigh = module.np.linalg.eigh

    def capture_eigh(matrix, *args, **kwargs):
        _write_qhmatrix(case["output_file"], matrix)
        raise _MatrixCaptured

    module.np.linalg.eigh = capture_eigh
    try:
        module.ee_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            params,
            shots=0,
            method="pyscf",
            basis=cfg["basis"],
            unit=cfg["unit"],
            state_idx=0,
            r1r2_outfile=None,
            hamiltonian=hamiltonian,
            qubits=qubits,
            symmetry_roots=0,
        )
    except _MatrixCaptured:
        pass
    finally:
        module.np.linalg.eigh = original_eigh


def _capture_full_operator_matrix(case, cfg, hamiltonian, qubits, params):
    original_eigh = np.linalg.eigh

    def capture_eigh(matrix, *args, **kwargs):
        _write_qhmatrix(case["output_file"], matrix)
        raise _MatrixCaptured

    np.linalg.eigh = capture_eigh
    try:
        case["module"].ee_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            params,
            shots=0,
            method="pyscf",
            basis=cfg["basis"],
            unit=cfg["unit"],
            state_idx=0,
            r1r2_outfile=None,
            hamiltonian=hamiltonian,
            qubits=qubits,
            symmetry_roots=0,
        )
    except _MatrixCaptured:
        pass
    finally:
        np.linalg.eigh = original_eigh


def _run_case(case):
    amp_file = case["amp_file"]
    if not amp_file.exists():
        raise FileNotFoundError(f"Missing UCCSD output file: {amp_file}")

    cfg = _problem(case["distance_factor"])
    print(f"CASE={case['name']}")
    print(f"UCCSD_OUTPUT={amp_file}")

    hamiltonian, qubits, _ = casci.build_casci_hamiltonian_from_problem(
        cfg,
        n_excited=0,
        casci_output_path=None,
    )
    qubits = int(qubits)
    params = _load_params_from_t1_t2(amp_file, cfg, qubits)

    if case["exact"]:
        _capture_full_operator_matrix(case, cfg, hamiltonian, qubits, params)
    else:
        _capture_trotterized_matrix(case, cfg, hamiltonian, qubits, params)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Write CH+ qHMatrix files from saved UCCSD t1_t2 outputs."
    )
    parser.add_argument(
        "--case",
        choices=[case["name"] for case in _cases()],
        action="append",
        help="Run only the named case. Can be passed more than once.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cases = _cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["name"] in wanted]

    for case in cases:
        _run_case(case)


if __name__ == "__main__":
    main()
