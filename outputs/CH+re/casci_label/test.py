import numpy as np
from pyscf import gto, scf, mcscf, symm

def run_ch_plus_casci(bond_length=1.12):
    """
    Computes the CASCI eigenspectrum of CH+ at a given geometry
    and labels the states by their dominant spatial symmetry.
    """
    # 1. Define Molecule (Singlet CH+ has 6 electrons total)
    mol = gto.Mole()
    mol.atom = f'C 0 0 0; H 0 0 {bond_length}'
    mol.basis = '6-31g'
    mol.charge = 1
    mol.spin = 0
    mol.symmetry = 'C2v'  # Enforce point group symmetry (C2v)
    mol.verbose =4 
    mol.build()

    # 2. Run Mean-Field Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # 3. Setup CASCI (6 electrons, 6 active orbitals)
    # Total electrons = 6, Active electrons = 6 (Core = 0, Active = 6)
    ncas = 6
    nelecas = 6
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.kernel()
    # 4. Map C2v Irreps to Linear Molecule Notation
    # PySCF uses C2v: A1=0, B1=1, B2=2, A2=3
    # For a linear molecule along the Z axis:
    # A1 maps to Sigma+ (or Delta if mixed, but ground state configuration yields Sigma+)
    # B1/B2 are the degenerate components of Pi
    # A2 maps to Delta (along with one component from A1)
    irrep_labels_c2v = {0: 'A1', 1: 'B1', 2: 'B2', 3: 'A2'}
    
    all_states = []

    # 5. Solve FCIs separately for each C2v symmetry block to get the full spectrum
    for irrep_id, irrep_name in irrep_labels_c2v.items():
        # Determine the maximum possible number of determinants/roots in this sector
        # We try to request a large number (e.g., 50) to capture the full spectrum
        mc.fcisolver.nroots = 50 
        mc.fcisolver.wfnsym = irrep_id
        
        # Run CASCI for this specific symmetry block
        try:
            e_tot, civec = mc.kernel()
        except Exception:
            # Handle cases where fewer roots exist than requested
            continue

        # If only one root was found, make it iterable
        if isinstance(e_tot, float):
            e_tot = [e_tot]

        # Append state data (Energy, C2v Irrep)
        for root_idx, energy in enumerate(e_tot):
            all_states.append({
                'energy': energy,
                'c2v_irrep': irrep_name,
                'root_within_irrep': root_idx + 1
            })

    # 6. Sort all states across all irreps by energy
    all_states = sorted(all_states, key=lambda s: s['energy'])

    # 7. Map C2v Irreps to Linear Group Irreps for Presentation
    # Due to degeneracy, we look for pairings to resolve Pi and Delta states
    print(f"{'State #':<9}{'Total Energy (Eh)':<22}{'C2v Irrep':<12}{'Assigned Linear Irrep':<15}")
    print("-" * 60)
    print(all_states)

 
    for idx, state in enumerate(all_states):
        c2v = state['c2v_irrep']
        linear_irrep = "Unknown"
        
        if c2v in ['B1', 'B2']:
            linear_irrep = r"1^Pi"
        elif c2v == 'A2':
            linear_irrep = r"1^Delta"
        elif c2v == 'A1':
            # A1 can technically be 1^Sigma+ or the other component of 1^Delta.
            # For a quick assignment, we look if it is degenerate with an A2 state.
            is_delta = False
            for other in all_states:
                if other['c2v_irrep'] == 'A2' and np.isclose(other['energy'], state['energy'], atol=1e-5):
                    is_delta = True
                    break
            linear_irrep = r"1^Delta" if is_delta else r"1^Sigma+"

        print(f"{idx+1:<9}{state['energy']:<22.8f}{c2v:<12}{linear_irrep:<15}")

if __name__ == "__main__":
    # Run at an arbitrary geometry (e.g., R = 1.15 Angstroms)
    run_ch_plus_casci(bond_length=1.15)
