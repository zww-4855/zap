from pyscf import gto, scf, mcscf

mol = gto.Mole()
mol.atom = [('B', (0,0,0)),('H', (2.3289000000,0,0))]
mol.basis ='6-31g'
#mol.charge = 1
#mol.spin = 0
mol.unit = 'Bohr'
#mol.symmetry = 'c2v'
mol.symmetry = False
mol.build()
print(mol.topgroup)
print(mol.groupname)

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-12
mf.max_cycle = 2000
mf.verbose = 4
mf.kernel()

from pyscf.fci import cistring

norb = 6  # Number of orbitals
neleca = 3  # Number of alpha electrons

# Generate alpha strings (represented as integers)
alpha_strs = cistring.make_strings(range(norb), neleca)

irrep_labels_c2v = {0: 'A1', 1: 'B1', 2: 'B2', 3: 'A2'}
mc = mcscf.CASCI(mf,6,6)
mc.fcisolver.nroots = 400
#mc.fcisolver.wfnsym = 'B1'
mc.kernel()
print("Total energies:", mc.e_tot)
# for irrep_id, irrep_name in irrep_labels_c2v.items():
# 	mc.fcisolver.wfnsym = irrep_id
# 	mc.fcisolver.nroots = 10
# 	print(irrep_name)
# 	mc.kernel()
# 	print("Total energies:", mc.e_tot)

for k in range(mc.fcisolver.nroots):
	print("State",k)
	print("dE = ",(mc.e_tot[k] + 37.9175585622613)*27.211386246)
	for i in range(len(alpha_strs)):
		for j in range(len(alpha_strs)):
			if(abs(mc.ci[k][i,j]) > 0.1):
				print(mc.ci[k][i,j], f"{alpha_strs[i]:0{norb}b}", f"{alpha_strs[j]:0{norb}b}")

