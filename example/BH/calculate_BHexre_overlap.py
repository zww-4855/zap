import ast
import itertools
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp") / "matplotlib-codex"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import casci
import exc
import vqee

OVERLAP_REPORT_NAME = "casci_qsceom_overlaps"
LARGE_ERROR_AMPLITUDE_REPORT_NAME = "large_error_delta_state_amplitudes"
AMPLITUDE_THRESHOLD = 0.1
DELTA_LIKE_IRREPS = {"A1", "A2"}


def _case_configs():
    bh_re = 2.3289
    ch_re = 2.13713
    cases = []

    for tag, factor in [("re", 1.0), ("1.5re", 1.5), ("2re", 2.0)]:
        cases.append(
            {
                "name": f"BH{tag}-full",
                "run_file": f"example/BH/run_BHex{tag}.py" if tag != "re" else "example/BH/run_BHexre.py",
                "symbols": ["B", "H"],
                "bond_label": tag,
                "bond_length": factor * bh_re,
                "charge": 0,
                "ansatz": "exact",
                "ansatz_label": "full",
                "output_dir": PROJECT_ROOT / "outputs" / f"BH{tag}" / "BH Full operator",
            }
        )
        cases.append(
            {
                "name": f"BH{tag}-trotterized",
                "run_file": f"example/BH/run_BH{tag}.py" if tag != "re" else "example/BH/run_BHre.py",
                "symbols": ["B", "H"],
                "bond_label": tag,
                "bond_length": factor * bh_re,
                "charge": 0,
                "ansatz": "trotterized",
                "ansatz_label": "trot",
                "output_dir": PROJECT_ROOT / "outputs" / f"BH{tag}" / "BH Trotterized",
            }
        )

    for tag, factor in [("re", 1.0), ("1.5re", 1.5), ("2re", 2.0)]:
        cases.append(
            {
                "name": f"CH+{tag}-full",
                "run_file": f"example/CH+/run_CH+ex{tag}.py" if tag != "re" else "example/CH+/run_CH+exre.py",
                "symbols": ["C", "H"],
                "bond_label": tag,
                "bond_length": factor * ch_re,
                "charge": 1,
                "ansatz": "exact",
                "ansatz_label": "full",
                "output_dir": PROJECT_ROOT / "outputs" / f"CH+{tag}" / "CH+ Full operator",
            }
        )
        cases.append(
            {
                "name": f"CH+{tag}-trotterized",
                "run_file": f"example/CH+/run_CH+{tag}.py" if tag != "re" else "example/CH+/run_CH+re.py",
                "symbols": ["C", "H"],
                "bond_label": tag,
                "bond_length": factor * ch_re,
                "charge": 1,
                "ansatz": "trotterized",
                "ansatz_label": "trot",
                "output_dir": PROJECT_ROOT / "outputs" / f"CH+{tag}" / "CH+ Trotterized",
            }
        )

    return cases


def _problem_from_case(case):
    import numpy as np

    return {
        "symbols": case["symbols"],
        "geometry": np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, float(case["bond_length"])]],
            dtype=float,
        ),
        "active_electrons": 6,
        "active_orbitals": 6,
        "charge": int(case["charge"]),
        "basis": "6-31g",
        "unit": "bohr",
    }


def _molecule_label(case):
    label = "".join(case["symbols"])
    if int(case["charge"]) > 0:
        label += "+" * int(case["charge"])
    elif int(case["charge"]) < 0:
        label += "-" * abs(int(case["charge"]))
    return label


def _bond_length_label(case):
    return str(case["bond_label"])


def _case_report_filename(case, report_name):
    return (
        f"{_molecule_label(case)}_{_bond_length_label(case)}_"
        f"{case['ansatz_label']}_{report_name}.xlsx"
    )


def _occ_to_index(occ, n_qubits, wire0_is_msb):
    idx = 0
    for w in occ:
        bit = (n_qubits - 1 - int(w)) if wire0_is_msb else int(w)
        idx |= 1 << bit
    return idx


def _fixed_spin_determinant_indices(active_electrons, active_orbitals, wire0_is_msb):
    n_qubits = 2 * int(active_orbitals)
    n_alpha = int(active_electrons) // 2
    n_beta = int(active_electrons) - n_alpha
    alpha_orbs = list(range(0, n_qubits, 2))
    beta_orbs = list(range(1, n_qubits, 2))

    indices = []
    determinants = []
    for alpha_occ in itertools.combinations(alpha_orbs, n_alpha):
        for beta_occ in itertools.combinations(beta_orbs, n_beta):
            occ = tuple(sorted(alpha_occ + beta_occ))
            determinants.append(occ)
            indices.append(_occ_to_index(occ, n_qubits, wire0_is_msb))
    return determinants, indices


def _parse_complex(text):
    text = text.strip()
    try:
        return complex(ast.literal_eval(text))
    except Exception:
        return complex(text)


def _parse_amplitudes(path, singles, doubles):
    import numpy as np

    label_to_slot = {}
    for idx, (i, a) in enumerate(singles):
        label_to_slot[f"{a}^ {i}"] = (idx, 1.0)

    offset = len(singles)
    for didx, (i, j, a, b) in enumerate(doubles):
        a2, b2, s_ab = vqee._canon_pair(a, b)
        i2, j2, s_ij = vqee._canon_pair(i, j)
        label = f"{a2}^ {b2}^ {i2} {j2}"
        label_to_slot[label] = (offset + didx, s_ab * s_ij)

    params = np.full(len(singles) + len(doubles), np.nan, dtype=float)
    line_re = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = line_re.match(line)
        if match is None:
            continue
        label = match.group(1).strip()
        if label not in label_to_slot:
            continue
        slot, display_sign = label_to_slot[label]
        value = float(_parse_complex(match.group(2)).real)
        params[slot] = value / display_sign

    missing = np.flatnonzero(np.isnan(params))
    if missing.size:
        raise RuntimeError(f"Missing {missing.size} amplitudes from {path}: {missing[:10]}")
    return params


def _parse_qsceom_energy_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = next(handle, None)
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            soln_idx, energy, dominant_irrep = stripped.split("\t")
            rows.append(
                {
                    "soln_idx": int(soln_idx),
                    "energy": float(energy),
                    "dominant_irrep": dominant_irrep,
                }
            )
    if not rows:
        raise RuntimeError(f"No qscEOM rows found in {path}")
    return rows


def _parse_r_vector(path):
    values = []
    line_re = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = line_re.match(line)
        if match is None:
            continue
        try:
            values.append(_parse_complex(match.group(2)))
        except ValueError:
            continue
    return values


def _parse_labeled_amplitudes(path):
    values = []
    line_re = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = line_re.match(line)
        if match is None:
            continue
        label = match.group(1).strip()
        try:
            coeff = _parse_complex(match.group(2))
        except ValueError:
            continue
        values.append((label, coeff))
    return values


def _t_amplitude_kind(label):
    return "T2" if label.count("^") == 2 else "T1"


def _r_amplitude_kind(label):
    if label == "reference":
        return "R0"
    return "R2" if ";" in label else "R1"


def _load_qsceom_vectors(output_dir, energy_rows, det_count):
    import numpy as np

    vectors = []
    for row in energy_rows:
        path = output_dir / "r_vectors" / f"r1_r2_soln{row['soln_idx']}"
        values = _parse_r_vector(path)
        if len(values) != det_count:
            raise RuntimeError(
                f"{path} has {len(values)} coefficients, expected {det_count}."
            )
        vec = np.asarray(values, dtype=complex)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise RuntimeError(f"{path} has zero norm.")
        vectors.append(vec / norm)
    return np.column_stack(vectors)


def _selected_qsceom_soln(output_dir, q_vectors, energy_rows):
    import numpy as np

    selected_path = output_dir / "r1_r2.txt"
    if not selected_path.exists():
        return None

    selected = np.asarray(_parse_r_vector(selected_path), dtype=complex)
    if selected.size != q_vectors.shape[0]:
        return None
    norm = np.linalg.norm(selected)
    if norm == 0:
        return None
    selected = selected / norm
    scores = np.abs(q_vectors.conj().T @ selected)
    best = int(np.argmax(scores))
    if float(scores[best]) < 1.0 - 1.0e-8:
        return None
    return int(energy_rows[best]["soln_idx"])


def _sparse_creation_annihilation(n_modes, wire_order, wire0_is_msb):
    import numpy as np
    from scipy import sparse

    dim = 1 << n_modes
    wire_to_pos = {w: idx for idx, w in enumerate(wire_order)}

    def bitpos(mode):
        pos = wire_to_pos[mode]
        return (n_modes - 1 - pos) if wire0_is_msb else pos

    parity_masks = []
    mask = 0
    for p in range(n_modes):
        parity_masks.append(mask)
        mask |= 1 << bitpos(p)

    annihilation = []
    creation = []
    for p in range(n_modes):
        a_rows = []
        a_cols = []
        a_data = []
        adag_rows = []
        adag_cols = []
        adag_data = []

        bp = bitpos(p)
        flip = 1 << bp
        parity_mask = parity_masks[p]
        for basis in range(dim):
            parity = bin(basis & parity_mask).count("1") & 1
            sign = -1.0 if parity else 1.0
            occupied = (basis >> bp) & 1
            if occupied:
                a_rows.append(basis ^ flip)
                a_cols.append(basis)
                a_data.append(sign)
            else:
                adag_rows.append(basis | flip)
                adag_cols.append(basis)
                adag_data.append(sign)

        annihilation.append(
            sparse.csr_matrix(
                (np.asarray(a_data), (a_rows, a_cols)),
                shape=(dim, dim),
                dtype=complex,
            )
        )
        creation.append(
            sparse.csr_matrix(
                (np.asarray(adag_data), (adag_rows, adag_cols)),
                shape=(dim, dim),
                dtype=complex,
            )
        )
    return annihilation, creation


def _build_ucc_generator(params, singles, doubles, n_qubits, wire_order, wire0_is_msb):
    from scipy import sparse

    annihilation, creation = _sparse_creation_annihilation(
        n_qubits, wire_order, wire0_is_msb
    )
    dim = 1 << n_qubits
    generator = sparse.csr_matrix((dim, dim), dtype=complex)

    for amp, (i, a) in zip(params[: len(singles)], singles):
        if abs(float(amp)) < 1.0e-14:
            continue
        term = creation[a] @ annihilation[i]
        generator = generator + float(amp) * (term - term.conj().T)

    for amp, (i, j, a, b) in zip(params[len(singles) :], doubles):
        if abs(float(amp)) < 1.0e-14:
            continue
        term = creation[a] @ creation[b] @ annihilation[j] @ annihilation[i]
        generator = generator + float(amp) * (term - term.conj().T)

    return generator.tocsr()


def _apply_exact_ucc(
    params, singles, doubles, n_qubits, wire_order, wire0_is_msb, qsc_seed
):
    from scipy.sparse.linalg import expm_multiply

    generator = _build_ucc_generator(
        params, singles, doubles, n_qubits, wire_order, wire0_is_msb
    )
    return expm_multiply(generator, qsc_seed)


def _apply_trotterized_ucc(params, singles, doubles, n_qubits, qsc_seed):
    import numpy as np
    import pennylane as qml

    wires = list(range(n_qubits))
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    init_state = np.zeros(n_qubits, dtype=int)
    ops = qml.UCCSD.compute_decomposition(
        np.asarray(params, dtype=float),
        wires=wires,
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=init_state,
        n_repeats=1,
    )[1:]

    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(seed_state):
        qml.StatePrep(seed_state, wires=wires)
        for op in ops:
            qml.apply(op)
        return qml.state()

    states = np.zeros_like(qsc_seed, dtype=complex)
    for col in range(qsc_seed.shape[1]):
        states[:, col] = np.asarray(circuit(qsc_seed[:, col]), dtype=complex)
    return states


def _apply_case_ucc(
    case, params, singles, doubles, n_qubits, wire_order, wire0_is_msb, qsc_seed
):
    if case["ansatz"] == "exact":
        return _apply_exact_ucc(
            params, singles, doubles, n_qubits, wire_order, wire0_is_msb, qsc_seed
        )
    if case["ansatz"] == "trotterized":
        return _apply_trotterized_ucc(params, singles, doubles, n_qubits, qsc_seed)
    raise ValueError(f"Unknown ansatz for {case['name']}: {case['ansatz']}")


def _casci_degeneracy_groups(energies, tol=1.0e-8):
    group_ids = []
    groups = []
    for idx, energy in enumerate(energies):
        if not groups or abs(float(energy) - groups[-1]["energy"]) > tol:
            groups.append({"energy": float(energy), "indices": []})
        groups[-1]["indices"].append(idx)
        group_ids.append(len(groups) - 1)
    return group_ids, groups


def _excel_column_name(col_idx):
    name = ""
    col_idx = int(col_idx) + 1
    while col_idx:
        col_idx, rem = divmod(col_idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell(value, row_idx, col_idx):
    ref = f"{_excel_column_name(col_idx)}{row_idx + 1}"
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return (
        f'<c r="{ref}" t="inlineStr"><is><t>'
        f"{escape(str(value))}"
        f"</t></is></c>"
    )


def _write_xlsx(path, headers, rows, sheet_name="overlaps"):
    sheet_rows = []
    for row_idx, row in enumerate([headers] + rows):
        cells = "".join(_xlsx_cell(value, row_idx, col_idx) for col_idx, value in enumerate(row))
        sheet_rows.append(f'<row r="{row_idx + 1}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="'
        + escape(sheet_name)
        + '" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", root_rels_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _problem_key(case):
    return (
        tuple(case["symbols"]),
        round(float(case["bond_length"]), 12),
        int(case["charge"]),
    )


def _build_casci_reference(case):
    import numpy as np
    import pennylane as qml

    cfg = _problem_from_case(case)

    print(f"Building CASCI reference for {case['name']}...")
    hamiltonian, n_qubits, _ = casci.build_casci_hamiltonian_from_problem(
        cfg,
        n_excited=1,
        casci_output_path=None,
    )

    wire_order = list(range(n_qubits))
    wire0_is_msb = vqee._wire0_is_msb(qml, wire_order)
    singles, doubles = qml.qchem.excitations(cfg["active_electrons"], n_qubits)

    qsc_det_list = exc.inite(cfg["active_electrons"], n_qubits)

    print("Diagonalizing CASCI Hamiltonian in the fixed Nalpha/Nbeta sector...")
    h_sparse = hamiltonian.sparse_matrix(wire_order=wire_order).tocsr()
    casci_dets, casci_indices = _fixed_spin_determinant_indices(
        cfg["active_electrons"], cfg["active_orbitals"], wire0_is_msb
    )
    h_casci = h_sparse[casci_indices, :][:, casci_indices].toarray()
    h_casci = 0.5 * (h_casci + h_casci.conj().T)
    casci_energies, casci_vecs = np.linalg.eigh(h_casci)
    order = np.argsort(casci_energies.real)
    casci_energies = casci_energies[order].real
    casci_vecs = casci_vecs[:, order]
    casci_excited_offset = 1
    casci_excited_energies = casci_energies[casci_excited_offset:]
    casci_excited_vecs = casci_vecs[:, casci_excited_offset:]
    excited_group_ids, excited_groups = _casci_degeneracy_groups(casci_excited_energies)

    return {
        "cfg": cfg,
        "n_qubits": n_qubits,
        "wire_order": wire_order,
        "wire0_is_msb": wire0_is_msb,
        "singles": singles,
        "doubles": doubles,
        "qsc_det_list": qsc_det_list,
        "casci_indices": casci_indices,
        "casci_excited_offset": casci_excited_offset,
        "casci_excited_energies": casci_excited_energies,
        "casci_excited_vecs": casci_excited_vecs,
        "excited_group_ids": excited_group_ids,
        "excited_groups": excited_groups,
    }


def _write_case_reports(case, reference):
    import numpy as np

    output_dir = case["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_file = output_dir / _case_report_filename(case, OVERLAP_REPORT_NAME)
    large_error_amplitude_file = output_dir / _case_report_filename(
        case, LARGE_ERROR_AMPLITUDE_REPORT_NAME
    )

    required_paths = [
        output_dir / "t1_t2.txt",
        output_dir / "qsceom_energy",
        output_dir / "r_vectors",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"{case['name']} is missing required saved outputs: {missing_list}")

    cfg = reference["cfg"]
    n_qubits = reference["n_qubits"]
    wire_order = reference["wire_order"]
    wire0_is_msb = reference["wire0_is_msb"]
    singles = reference["singles"]
    doubles = reference["doubles"]
    qsc_det_list = reference["qsc_det_list"]

    print(f"Processing {case['name']} from {case['run_file']}...")
    params = _parse_amplitudes(output_dir / "t1_t2.txt", singles, doubles)
    energy_rows = _parse_qsceom_energy_rows(output_dir / "qsceom_energy")
    q_vectors = _load_qsceom_vectors(output_dir, energy_rows, len(qsc_det_list))

    print(f"Applying {case['ansatz']} UCC unitary to qscEOM determinant expansions...")
    qsc_seed = np.zeros((1 << n_qubits, q_vectors.shape[1]), dtype=complex)
    for det_idx, occ in enumerate(qsc_det_list):
        basis_idx = _occ_to_index(occ, n_qubits, wire0_is_msb)
        qsc_seed[basis_idx, :] = q_vectors[det_idx, :]

    qsc_states = _apply_case_ucc(
        case, params, singles, doubles, n_qubits, wire_order, wire0_is_msb, qsc_seed
    )

    qsc_states_in_casci_sector = qsc_states[reference["casci_indices"], :]
    overlaps = reference["casci_excited_vecs"].conj().T @ qsc_states_in_casci_sector
    abs_overlaps = np.abs(overlaps)
    overlap_sq = abs_overlaps**2

    group_overlap_sq = np.zeros((len(reference["excited_groups"]), q_vectors.shape[1]), dtype=float)
    for group_id, group in enumerate(reference["excited_groups"]):
        group_overlap_sq[group_id, :] = np.sum(overlap_sq[group["indices"], :], axis=0)

    headers = [
        "qsc_soln_idx",
        "qsc_energy_hartree",
        "casci_state_idx",
        "casci_energy_hartree",
        "dominant_irrep",
        "Energy error",
        "overlap",
    ]
    output_rows = []
    for col, row in enumerate(energy_rows):
        assigned_excited_idx = int(row["soln_idx"])
        if assigned_excited_idx >= len(reference["casci_excited_energies"]):
            continue
        assigned_group_id = reference["excited_group_ids"][assigned_excited_idx]
        assigned_overlap_sq = float(group_overlap_sq[assigned_group_id, col])
        assigned_overlap = float(np.sqrt(max(assigned_overlap_sq, 0.0)))
        assigned_state_idx = assigned_excited_idx + reference["casci_excited_offset"]
        qsc_energy = float(row["energy"])
        casci_energy = float(reference["casci_excited_energies"][assigned_excited_idx])
        output_rows.append(
            [
                int(row["soln_idx"]),
                qsc_energy,
                assigned_state_idx,
                casci_energy,
                row["dominant_irrep"],
                qsc_energy - casci_energy,
                assigned_overlap,
            ]
        )

    print(f"Writing assigned overlaps to {overlap_file}...")
    _write_xlsx(overlap_file, headers, output_rows)

    selected_delta_rows = sorted(
        [row for row in output_rows if row[4] in DELTA_LIKE_IRREPS],
        key=lambda row: abs(float(row[5])),
        reverse=True,
    )[:3]

    t_amplitudes = [
        (_t_amplitude_kind(label), label, coeff)
        for label, coeff in _parse_labeled_amplitudes(output_dir / "t1_t2.txt")
        if abs(coeff) > AMPLITUDE_THRESHOLD
    ]

    amplitude_headers = [
        "qsc_soln_idx",
        "qsc_energy_hartree",
        "casci_state_idx",
        "casci_energy_hartree",
        "dominant_irrep",
        "Energy error",
        "amplitude_kind",
        "operator",
        "amplitude",
    ]
    amplitude_rows = []
    for selected in selected_delta_rows:
        for kind, label, coeff in t_amplitudes:
            amplitude_rows.append(
                [
                    int(selected[0]),
                    float(selected[1]),
                    int(selected[2]),
                    float(selected[3]),
                    selected[4],
                    float(selected[5]),
                    kind,
                    label,
                    float(coeff.real),
                ]
            )

        r_path = output_dir / "r_vectors" / f"r1_r2_soln{int(selected[0])}"
        for label, coeff in _parse_labeled_amplitudes(r_path):
            if abs(coeff) <= AMPLITUDE_THRESHOLD:
                continue
            amplitude_rows.append(
                [
                    int(selected[0]),
                    float(selected[1]),
                    int(selected[2]),
                    float(selected[3]),
                    selected[4],
                    float(selected[5]),
                    _r_amplitude_kind(label),
                    label,
                    float(coeff.real),
                ]
            )

    print(f"Writing large-error Delta-like amplitudes to {large_error_amplitude_file}...")
    _write_xlsx(large_error_amplitude_file, amplitude_headers, amplitude_rows, "amplitudes")

    print(f"Wrote {len(output_rows)} overlap rows and {len(amplitude_rows)} amplitude rows.")
    print("Selected Delta-like roots for amplitude report:")
    for row in selected_delta_rows:
        print(
            f"  qsc soln {int(row[0]):2d} ({row[4]}): "
            f"error={float(row[5]):+.8f} Ha"
        )
    return output_rows, amplitude_rows, selected_delta_rows


def _parse_args():
    import argparse

    cases = _case_configs()
    parser = argparse.ArgumentParser(
        description="Write CASCI/qscEOM overlap and large-error Delta amplitude reports."
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case["name"] for case in cases],
        help="Case name to run. Repeat for multiple cases. Default: all BH and CH+ cases.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    all_cases = _case_configs()
    if args.case:
        selected = set(args.case)
        cases = [case for case in all_cases if case["name"] in selected]
    else:
        cases = all_cases

    reference_cache = {}
    completed = []
    for case in cases:
        key = _problem_key(case)
        if key not in reference_cache:
            reference_cache[key] = _build_casci_reference(case)
        output_rows, amplitude_rows, selected_delta_rows = _write_case_reports(
            case, reference_cache[key]
        )
        completed.append((case, len(output_rows), len(amplitude_rows), selected_delta_rows))

    print("Completed report generation:")
    for case, n_overlap, n_amplitude, selected_delta_rows in completed:
        roots = ", ".join(str(int(row[0])) for row in selected_delta_rows) or "none"
        print(
            f"  {case['name']}: {n_overlap} overlaps, "
            f"{n_amplitude} amplitudes, selected roots {roots}"
        )


if __name__ == "__main__":
    main()
