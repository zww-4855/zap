import numpy as np
from pyscf import gto, scf, mcscf, fci
bond_length=[2.1371300000]
def run_ch_plus_casci_robust(bond_length=2.1371300000):
    mol = gto.Mole()
    mol.unit = "B"
    mol.atom = f'C 0 0 0; H 0 0 4.2742600000'
    mol.basis = '6-31g'
    mol.charge = 1
    mol.spin = 0
    mol.symmetry = "C2v"  # Enforce C2v symmetry
    mol.verbose = 4 
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    # 6 active orbitals, 6 active electrons
    ncas = 6
    nelecas = 6 
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.frozen = list(range(13,23)) 
    all_states = []


    all_states = []
    
    for irrep_name in ['A1', 'B1', 'B2', 'A2']:
    
        mc = mcscf.CASCI(mf, ncas, nelecas)
    
        mc.fcisolver.wfnsym = irrep_name
        mc.fcisolver.nroots = 20 
    
        try:
            e_tot, civec = mc.kernel()
    
            if np.isscalar(e_tot):
                e_tot = [e_tot]
    
            for energy in e_tot:
                all_states.append({
                    'energy': energy,
                    'c2v_irrep': irrep_name
                })
    
        except Exception as e:
            print(f"Failed for {irrep_name}: {e}")


    sys.exit()

    irrep_labels_c2v = {0: 'A1', 1: 'B1', 2: 'B2', 3: 'A2'}

    for irrep_id, irrep_name in irrep_labels_c2v.items():
        mc.fcisolver.wfnsym = irrep_id
        
        # Start by trying to get up to 30 roots
        requested_roots = 30
        success = False
        
        while requested_roots > 0 and not success:
            mc.fcisolver.nroots = requested_roots
            try:
                e_tot, civec = mc.kernel()
                success = True
                
                # If only a single root is returned, make it a list
                if isinstance(e_tot, (float, np.float64)):
                    e_tot = [e_tot]
                    
                for root_idx, energy in enumerate(e_tot):
                    all_states.append({
                        'energy': energy,
                        'c2v_irrep': irrep_name
                    })
            except Exception as e:
                # If it fails due to too many requested roots, lower the count and retry
                if "Linear dependency" in str(e) or "Too many roots" in str(e) or requested_roots > 1:
                    requested_roots -= 2  # Step down and retry
                else:
                    # If it's a different error or we hit 0, move to the next irrep
                    break

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
