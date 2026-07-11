import read_info as read_info
import numpy as np


def helper():
    print("helper function called")
    ## ------------------------------------------------------------
    ## BUILD OUT D3 DENOMINATORS
    fock_file = "/Users/zwu/Desktop/work/bkup/zap/outputs/CH+re/CH+ Full operator/fock.txt"
    nocc,nvirt,mo_energies,oei = read_info.read_fock(fock_file)
    eps_a = np.asarray(mo_energies)
    eps_b = np.asarray(mo_energies)
    eps = np.append(eps_a, eps_b)
    eps=np.sort(eps)

    n=np.newaxis
    o=slice(None,nocc)
    v=slice(nocc,None)
    D3 = read_info.D3denomSlow(eps,o,v,n)


    print("Number of occupied orbitals:", nocc)
    print("Number of virtual orbitals:", nvirt)
    print("Molecular orbital energies:", mo_energies)
    print("One-electron integrals (OEI):", oei)
    print("D3 denominators:", D3)
    #sys.exit()

    print("helper:",2*len(mo_energies))
    #sys.exit()

    ## ------------------------------------------------------------
    ## READ IN TWO-ELECTRON INTEGRALS (TEI) AND UCC G.S. AMPLITUDES
    ## ------------------------------------------------------------
    in_file = "/Users/zwu/Desktop/work/bkup/zap/outputs/CH+re/CH+ Full operator/two_elec.txt"
    tei = read_info.read_tei(in_file, 2*len(mo_energies))
    tamp_infile = "/Users/zwu/Desktop/work/bkup/zap/outputs/CH+re/CH+ Full operator/t1_t2.txt"
    t2amps, t1amps = read_info.read_tamps(tamp_infile,nocc,nvirt)

    print("Two-electron integrals (TEI):",tei)

    #mp2_energy = read_info.mp2_energy(nocc,nvirt,mo_energies,tei)
    #print("MP2 energy:", mp2_energy)

    return nocc,nvirt,tei,t2amps,t1amps,D3,o,v,mo_energies
