import itertools
import pickle
from pyscf.lib import logger
from pyscf.data import nist
import pycc
import pyscf
from pyscf import gto, mp, mcscf


dist=[1.1285000000]
dist=[2.2618409850]
for num in dist:
    atomString=f'H 0. 0. 0.0; C 0.0 0.0 {num}'
    value=[]
    basis='6-31G'
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        charge = 1,
#        symmetry =True,
        basis=basis)
    mf = mol.RHF()

    mf.run()
#    from pyscf import mp
#    mymp = mp.RMP2(mf).run()

#    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)

#    ncas, nelecas = (6,6)
#    mycas = mcscf.CASCI(mf, ncas, nelecas)
#    mycas.frozen = 2
#    mycas.natorb = True

#    mycas.kernel()

    bkgrd_infile='fock.txt'
    tamp_infile='t1_t2.txt'
    tei_infile='two_elec.txt'

    r_vec = 'r_vectors/OUTPUT'
    pycc.XaccCorrection('EOMT',bkgrd_infile,tamp_infile,tei_infile,ref="spin-orbitalEOM",eom_infile=r_vec) #xaccObj)
