import numpy as np


def read_Hbar_matrix_from_txt(
    filename,
    dim=117,
    dtype=float,
    check_square=True,
    check_symmetric=True,
    symmetrize=False,
    atol=1e-10,
):
    """
    Read a whitespace- or tab-delimited matrix from a text file.

    Parameters
    ----------
    filename : str
        Path to the text file containing the matrix.
    dim : int
        Expected matrix dimension. For your case, dim = 117.
    dtype : type
        Data type for the matrix entries.
    check_square : bool
        If True, require the matrix to be square.
    check_symmetric : bool
        If True, check whether the matrix is Hermitian/symmetric.
    symmetrize : bool
        If True, replace A by 0.5 * (A + A.T.conj()).
        Useful if the input has tiny numerical asymmetries.
    atol : float
        Absolute tolerance for the symmetry check.

    Returns
    -------
    A : ndarray, shape (dim, dim)
        Matrix read from the text file.
    """

    # delimiter=None means "split on any whitespace", including spaces and tabs
    A = np.loadtxt(filename, dtype=dtype, delimiter=None)

    # In case the file was stored as one long vector of dim*dim numbers
    if A.ndim == 1:
        if A.size != dim * dim:
            raise ValueError(
                f"File contains {A.size} numbers, but expected {dim * dim} "
                f"for a {dim} x {dim} matrix."
            )
        A = A.reshape((dim, dim))

    if check_square and A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix is not square. Got shape {A.shape}.")

    if A.shape != (dim, dim):
        raise ValueError(f"Expected matrix shape {(dim, dim)}, but got {A.shape}.")

    if check_symmetric:
        if not np.allclose(A, A.T.conj(), atol=atol):
            max_asym = np.max(np.abs(A - A.T.conj()))

            if symmetrize:
                print(
                    f"Warning: matrix was not exactly symmetric. "
                    f"max |A - A.T| = {max_asym:.3e}. Symmetrizing."
                )
                A = 0.5 * (A + A.T.conj())
            else:
                raise ValueError(
                    f"Matrix is not symmetric/Hermitian within atol={atol}. "
                    f"max |A - A.T| = {max_asym:.3e}. "
                    f"Set symmetrize=True if this is only roundoff noise."
                )

    return A


def subtract_ground_state_energy_from_diagonal(Hbar, E_uccsd, copy=True):
    """
    Subtract the ground-state UCCSD energy from the diagonal elements of Hbar.

    This performs

        Hbar_shifted = Hbar - E_uccsd * I

    Parameters
    ----------
    Hbar : ndarray
        Effective Hamiltonian matrix.

    E_uccsd : float
        Ground-state UCCSD energy.

    copy : bool
        If True, return a shifted copy of Hbar.
        If False, modify Hbar in place.

    Returns
    -------
    Hbar_shifted : ndarray
        Matrix whose diagonal has been shifted by -E_uccsd.
    """

    Hbar = np.array(Hbar, copy=copy)

    if Hbar.ndim != 2 or Hbar.shape[0] != Hbar.shape[1]:
        raise ValueError(f"Hbar must be a square matrix. Got shape {Hbar.shape}.")

    E_uccsd = float(E_uccsd)

    diag_idx = np.diag_indices_from(Hbar)
    Hbar[diag_idx] -= E_uccsd

    return Hbar



def read_tei(tei_infile,dim):
    """
    Reads two-electron integrals from the input file and stores them in a 4D numpy array.
    accessible as a class attribute

    :param tei_infile: Path to the input file containing two-electron integrals.

    :return: None
    """

    tei_tmp={}
    tei=np.zeros((dim,dim,dim,dim))
    with open(tei_infile,'r') as f:
        for line in f:
            tei_key=float(line.split()[-1])
            index_list=line.split()
            operator_list=[]
            for operator in range(4): # max T2, min T1
                if index_list[operator] == '|':
                    break
                operator_list.append(int(index_list[operator].strip('^')))
            #print('op list/key:',operator_list,tei_key,operator_list[0])
            idx=str(operator_list[0])+','+str(operator_list[1])+','+str(operator_list[2])+','+str(operator_list[3])
            a=operator_list[0]
            b=operator_list[1]
            c=operator_list[2]
            d=operator_list[3]
            tei[a,b,d,c]=tei_key

            tei_tmp.update({tei_key:operator_list})
        return tei
    

def read_fock(bkgrd_infile):
    """
    Reads pertinent background information, such as the number of occupied and virtual orbitals,
    as well as the molecular orbital energies from the specified input file.

    :param bkgrd_infile: Path to the background input file.

    :param ref: specifies whether or not we are pursuing corrections w.r.t. 'spatial' or
                'spin-orbital' (s)
    :return: None
    """
    with open(bkgrd_infile,'r') as f:
        lines=f.readlines()
    nocc=2*int(lines[1].strip().split()[-1])
    nvirt=2*int(lines[2].strip().split()[-1])
    print('FXN read_bkgrd info:',nocc,nvirt)
    tmp_energies=lines[3].strip().split()[-1]


    energy_line = lines[3]
    energy_str = energy_line.split(':', 1)[1].strip()
    energy_str = energy_str.strip('[]')


    mo_energies=[]
    mo_energies = [float(x) for x in energy_str.split()]

    # Add mo energy info to class variable storing oei
    eps = np.append(mo_energies,mo_energies)
    eps = np.sort(eps)
    oei=np.diag(eps)
    print('eps and mo_energies:',mo_energies,nocc,nvirt)
    return nocc,nvirt,mo_energies,oei


def mp2_energy(nocc,nvirt,mo_energies,tei):
    n=np.newaxis
    o=slice(None,nocc)
    v=slice(nocc,None)
    print(mo_energies,type(mo_energies[0]),nocc,o,v)
    eps_a = np.asarray(mo_energies)
    eps_b = np.asarray(mo_energies)
    print("eps a and b:",eps_a,eps_b,o,v)
    eps = np.append(eps_a, eps_b)
    eps=np.sort(eps)
    print("eps:",eps)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    print("e-abij:",e_abij)
    t2=e_abij*tei[v,v,o,o]
    print("test t2:",t2)
    #sys.exit()
    mp2E=0.250000000000000 * np.einsum('jiab,abji',tei[o, o, v, v], t2)
    return mp2E

def D3denomSlow(epsaa,occ_aa,virt_aa,n):
    D3 = 1.0/(
            -epsaa[virt_aa,n,n,n,n,n]
            -epsaa[n,virt_aa,n,n,n,n]
            -epsaa[n,n,virt_aa,n,n,n]
            +epsaa[n,n,n,occ_aa,n,n]
            +epsaa[n,n,n,n,occ_aa,n]
            +epsaa[n,n,n,n,n,occ_aa] )
    D3=D3.transpose(3,4,5,0,1,2)
    print("shapes:",np.shape(D3),occ_aa,virt_aa,n,epsaa,np.shape(epsaa))
    #sys.exit()
    return D3



def read_tamps(tamp_infile,nocc,nvirt):
    """
    Reads T1 and T2 CC amplitudes from the specified input file.
    and stores result in class attribute.

    :param tamp_infile: Path to the input file containing T1 and T2 amplitudes.

    :return: None
    """
    t2amp={}
    t1amp={}

    read_amps=True
    self_t2amps=np.zeros((nvirt,nvirt,nocc,nocc))
    self_t1amps=np.zeros((nvirt,nocc))
    print('reading tamp file:',tamp_infile)
    with open(tamp_infile,'r') as f:
        for line in f:
            if read_amps:#have to uncomment 109,13 for nospace
                #amp_key=float(line.split()[-1])#.strip('|'))
                print('line:',line)

                amp_key=float(line.split()[-1].strip('|'))
                #index_list=line.split()
                index_list=line.split()[:-1]
                index_list.append('|')
                index_list.append(amp_key)
                operator_list=[]
                for operator in range(6): # max T2, min T1
                    if index_list[operator] == '|':
                        break
                    operator_list.append(int(index_list[operator].strip('^')))
                if len(operator_list)==4: #dealing with t2amp
#                        t2amp.update({amp_key:operator_list})
                    a=operator_list[0]-nocc
                    b=operator_list[1]-nocc
                    i=operator_list[2]
                    j=operator_list[3]
                    self_t2amps[a,b,i,j]=amp_key
                    self_t2amps[b,a,i,j]= -1.0* amp_key
                    self_t2amps[a,b,j,i]= -1.0*amp_key
                    self_t2amps[b,a,j,i]=amp_key

                else: # dealing with t1amp
                    a=operator_list[0]-nocc
                    i=operator_list[1]
                    self_t1amps[a,i]=amp_key


    self_t2amps=-1.0*self_t2amps.transpose(2,3,0,1)  #self.t2amps.transpose(2,3,1,0)# ijab -> ijba convention ZWW 1/16/25
    self_t1amps=self_t1amps.transpose(1,0)
    return -1.0*self_t2amps,self_t1amps #following convention in pycc and how they print t2