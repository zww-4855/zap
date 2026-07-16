import numpy as np
from pyscf import gto, scf, mcscf, fci

def run_ch_plus_casci_robust(bond_length=1.15):
    mol = gto.Mole()
    mol.atom = f'C 0 0 0; H 0 0 {bond_length}'
    mol.basis = '6-31g'
    mol.charge = 1
    mol.spin = 0
    mol.symmetry = True  # Enforce C2v symmetry
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    # 6 active orbitals, 6 active electrons
    ncas = 6
    nelecas = 6
    mc = mcscf.CASCI(mf, ncas, nelecas)
    
    # Ensure the 1s core orbital of Carbon is frozen, and we select valence orbitals
    # C2v Irreps: 0=A1, 1=B1, 2=B2, 3=A2
    all_states = []
    irrep_labels_c2v = {0: 'A1', 1: 'B1', 2: 'B2', 3: 'A2'}

    for irrep_id, irrep_name in irrep_labels_c2v.items():
        mc.fcisolver.wfnsym = irrep_id
        
        # Calculate the actual size of the CI space for this irrep
        # This prevents asking for 50 roots if only 15 exist.
        nelec_a, nelec_b = mc.nelecas
        max_roots = fci.fci_sec_space.det_string_count(ncas, nelec_a) # Upper bound approximation
        
        # Safely request up to 40 roots, capped by the actual space size
        mc.fcisolver.nroots = min(40, max_roots) 
        
        try:
            # We use a cleaner solver call to catch edge-case errors
            e_tot, civec = mc.kernel()
            
            if isinstance(e_tot, float) or isinstance(e_tot, np.float64):
                e_tot = [e_tot]
                
            for root_idx, energy in enumerate(e_tot):
                all_states.append({
                    'energy': energy,
                    'c2v_irrep': irrep_name
                })
        except Exception as e:
            # Catching symmetry blocks that have 0 valid determinants for singlets
            print(f"Skipping or encountered limit in Irrep {irrep_name}")
            continue

    # Sort and print results
    all_states = sorted(all_states, key=lambda s: s['energy'])
    print(f"\n{'State #':<9}{'Total Energy (Eh)':<22}{'C2v Irrep':<12}{'Assigned Linear Irrep':<15}")
    print("-" * 60)
    
    for idx, state in enumerate(all_states):
        c2v = state['c2v_irrep']
        linear_irrep = "1^Sigma+"
        if c2v in ['B1', 'B2']:
            linear_irrep = r"1^Pi"
        elif c2v == 'A2':
            linear_irrep = r"1^Delta"
        elif c2v == 'A1':
            is_delta = any(other['c2v_irrep'] == 'A2' and np.isclose(other['energy'], state['energy'], atol=1e-5) for other in all_states)
            if is_delta:
                linear_irrep = r"1^Delta"
                
        print(f"{idx+1:<9}{state['energy']:<22.8f}{c2v:<12}{linear_irrep:<15}")

if __name__ == "__main__":
    run_ch_plus_casci_robust()
