# %% import modules
from numpy import array, linspace, pi, real, imag, sqrt, zeros, transpose, concatenate, dot, allclose, squeeze, log, abs, angle, exp, sum, eye
from kwant import plot as kplot
import matplotlib.pyplot as plt
from numpy.linalg import norm, det, eigh
from copy import deepcopy



import sys
import os
# Add the parent directory to Python path to find the modules folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.support_functions import sel_sites, Bi_partition_Hamiltonian, Pos_mat, dagger, plot_state, plot_system


# %% Function: local anyon basis for PBC system
def local_anyon_basis_pbc(msys, anyon_loop_indices, check = False, atol=1e-14):
    """
    Computes the transformation matrix T such that the transformed eigenstates of the Hamiltonian H'=T^dagger H T
    have local anyonic modes at the specified anyon_loop_indices.   
    The function assumes that the system has periodic boundary conditions (PBC).
    The function works by bi-partitioning the Hamiltonian into two parts A and B,
    where A contains the sites in the anyon loop and B contains the rest of the sites
    and finding the null space of the part of the Hamiltonian that couples A to B.
    The found vectors are then orthogonalized using Gram-Schmidt orthogonalization.
    The function also removes the global phase of the found vectors.
    Parameters
    ----------
    msys : object
        The system object containing the Hamiltonian and other parameters.
    anyon_loop_indices : list
        List of indices of the anyon loops where the local anyonic modes are to be found.
    check : bool, optional
        If True, the function will perform additional checks and plot the anyonic modes. The default is False.
    atol : float, optional
        The absolute tolerance for determining the number of zero modes. The default is 1e-14.
    Returns
    -------
    T_evec : np.array
        The transformation matrix T such that the transformed eigenstates of the Hamiltonian H'=T^dagger H T
        have local anyonic modes at the specified anyon_loop_indices.
    Raises
    ------
    AssertionError
        If the system does not have periodic boundary conditions (PBC). Please use function: "local_anyon_basis_obc" for OBC systems.   
    AssertionError
        If more than one not suitable vectors are found in the computations of local anyonic basis. no.of unsuitable vecs: {len(index_list)}
    AssertionError
        If the transformed vectors after gram-schmidt are not orthonormal: dagger(T) @ T != np.eye({N_max-N_min}), \n{dagger(T) @ T}
    """

    assert msys.syst_params['pbc'] == True, 'The system must have periodic boundary conditions for this function to work. Please use function: "local_anyon_basis_obc" for OBC systems.'

    h = msys.ham
    N_sites = msys.N_sites
    N_anyons = len(anyon_loop_indices)


    local_anyonic_states = zeros([N_sites, N_anyons], dtype = complex)
    index_list = []

    for index,loop_index in enumerate(anyon_loop_indices):
        A_indices, B_indices  = sel_sites(msys, loop_index)

        NA = len(A_indices)
        NB = len(B_indices)

        H_new, P = Bi_partition_Hamiltonian(h, A_indices, B_indices, check=check)
        P_inv = transpose(P)
        H_A = H_new[:NA, :NA]
        H_BA = H_new[NA:N_sites, :NA]

        M = concatenate([H_A, H_BA], axis=0)

        from scipy.linalg import null_space
        vec = null_space(M)

        if  not allclose(vec.shape, [NA,1]):
            index_list.append([index, loop_index])
            out_state = zeros([N_sites], dtype = complex)
        else:
            assert allclose(H_A @ vec, zeros([NA, 1])), f'H_A does not annihilate out_state. norm(H_A @ vec - Zero) = {norm(H_A @ vec)}'
            out_state = concatenate([squeeze(vec), squeeze(zeros([NB], dtype=complex))])
            assert  allclose(h @ (P_inv @ out_state), zeros([N_sites, 1])), f'Full Hamiltonian does not annihilate the found vector. norm(H @ vec - Zero) = {norm(h @ (P_inv) @ out_state)} '


        if check==True:
            assert allclose(H_BA @ vec, zeros([NB])), f'H_BA does not annihilate chosen state. norm(H_BA @ vec - Zero) = {norm(H_BA @ vec)}'
            H_A = H_new[:NA, :NA]
            fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
            state_color= log(abs(P_inv @ out_state)/norm(abs(P_inv @ out_state)))
            kplot(msys.syst, site_color=state_color, cmap = 'gist_heat_r', ax=ax[1])
            ax[1].collections[0].set_clim([-8,0])
            fig.colorbar(mappable=ax[1].collections[0], ax=ax[1])
            posmat = Pos_mat(msys.syst)
            for i in range(NA):
                ia = A_indices[i]
                ax[0].scatter(posmat[ia,1], posmat[ia,2], c='r')
            for i in range(NB):
                ib = B_indices[i]
                ax[0].scatter(posmat[ib,1], posmat[ib,2], c='b')
            plt.show()

        local_anyonic_states[:,index] = P_inv @ out_state

    assert len(index_list) <= 1, f'More than 1 not suitable vectors found in the computations of local anyonic basis: no.of unsuitable vecs: {len(index_list)}'
    
    #### Gram-Schmidt orthogonalization of the found vectors
        
    E_val, E_vec = eigh(h.toarray(), UPLO='L')
    N_min = sum(E_val<-1e-14)
    N_max = sum(E_val<1e-14)

    T_evec = E_vec
    for indices in index_list:
        index = indices[0]
        loop_index = indices[1]
        vec_in = sum(T_evec[:,N_min:N_max], axis = 1)
        new_indices = list(range(N_anyons))
        del new_indices[index]
        dot_list = [dot(local_anyonic_states[:, k], vec_in) for k in range(N_anyons)]
        projected_vecs = array([dot_list[k]  * local_anyonic_states[:,k] for k in new_indices])
        state = vec_in - sum(projected_vecs, axis =0)
        local_anyonic_states[:, index] = state/(norm(state))
    
    #### remove global phase
    for index in range(N_anyons):
        state = deepcopy(local_anyonic_states[:,index])

        state_phi = angle(state)
        state_phi[abs(state)<1e-10] = 0
        phi1 = max(state_phi)
        phi0 = min(state_phi)
        # print(f'phi1:{phi1/np.pi} phi0:{phi0/np.pi}')
        state = exp(-1j*phi1)*abs(state)*exp(1j*state_phi)
        
        state = state/(norm(state))


        local_anyonic_states[:, index] = state

        
    T_evec[:, N_min:N_max] = local_anyonic_states

    trans_vectors = T_evec[:, N_min:N_max]
    assert allclose(dagger(trans_vectors) @ trans_vectors, eye(N_max-N_min)), f'ERROR FOUND IN LOCAL ANYONIC STATES CONVERSION. The transformed vectors after gram-schmidt are not orthonormal: dagger(T) @ T != np.eye({N_max-N_min}), \n{dagger(trans_vectors) @ trans_vectors}'

    return T_evec

# %% function: many-body overlap calculation

#### function: nambu basis conversion
def nambu_basis_slow(E_vec, N_min, N_max):
    """
    Converts the single-particle eigenstates E_vec to the Nambu basis.
    The Nambu basis is defined as:
    \psi^-_n = 1/sqrt(2) * (\phi_n - i \phi_{n+N_zero/2})
    \psi^+_n = 1/sqrt(2) * (\phi_n + i \phi_{n+N_zero/2})
    where \phi_n are the original eigenstates and N_zero is the number of zero modes.
    The function assumes that the number of zero modes is even.
    Parameters
    ----------
    E_vec : np.array
        The single-particle eigenstates.
    N_min : int
        The minimum index of the zero modes.
    N_max : int
        The maximum index of the zero modes.
    Returns
    -------
    U : np.array
        The single-particle eigenstates in the Nambu basis.
    Raises
    ------
    AssertionError
        If the number of zero modes is not even.
    """ 
    U = deepcopy(E_vec)
    N_zero = N_max - N_min
    assert N_zero % 2 == 0, f'N_zero:{N_zero} is not even'
    N_zero2 = int(N_zero/2)
    for n in range(N_min, N_min+N_zero2):
        state_m = 1/sqrt(2)*(E_vec[:,n] - 1j* E_vec[:, n+N_zero2])
        state_p = 1/sqrt(2)*(E_vec[:,n] + 1j* E_vec[:, n+N_zero2])
        U[:,n] = state_m
        U[:,n+N_zero2] = state_p
    return U



#### many-body overlap for fermions
def many_body_product_state_overlap_fermionic(E_vec1, E_vec2, N_min, N_max, check=False):
    """
    Computes the fidelity measure of two many-body product-states \psi_1 and \psi_2 
    from their respective single-particle wavefunctions:
    
    \ket{\psi_1} = \sum_{i1,i2..iN} \varepsilon_{i1,i2,..,iN}/\sqrt{N!} \ket{a_i1} \ket{a_i2}...\ket{a_iN}
    \ket{\psi_2} = \sum_{j1,j2..jN} \varepsilon_{j1,j2,..,jN}/\sqrt{N!} \ket{b_j1} \ket{b_j2}...\ket{b_jN}
    
    Inner product  = \bra{\psi_1}\ket{\psi_2} = det(B)
    where B_ij = \bra{a_i}\ket{b_j}

    #### ASSUMES THAT EXACTLY ONE STATE IN THE DEGENARATE MANIFOLD IS PARTICIPATING IN THE PRODUCT STATE CREATION.

    Parameters
    ----------
    E_vec1 : np.array
        Single-particle states for constructing majorana many-body product state \ket{\psi_1}.
    E_vec2 : np.array
        Single-particle states for constructing majorana many-body product state \ket{\psi_2}.
    N_min: int
        Minimum index of single particle states in the degenerate manifold.
    N_max: int
        Maximum index of single particle states in the degenerate manifold.

    Returns
    -------
    X = [det(B^{ij})]
    """

    assert E_vec1.shape[0] == E_vec2.shape[0], f'Mismatch in dimension. E_vec1.shape[0]:{E_vec1.shape[0]} not equals E_vec2.shape[0]:{E_vec2.shape[0]}'

    if E_vec1.shape[1]!=E_vec2.shape[1]:
        print(f"Column lengths not equal of E_vec1 and E_vec2 in many body overlap calculation:{E_vec1.shape[1], E_vec2.shape[1]}")
        return 0
    else:        
        #### set_convention
        for i in range(N_min, N_max):
            overlap = dagger(E_vec1[:,i]) @ E_vec2[:,i]
            if real(overlap) < 0:
                E_vec2[:,i] = -E_vec2[:,i]
            new_overlap = dagger(E_vec1[:,i]) @ E_vec2[:,i]
            if check == True:
                print(f'Convention_check NAMBU: overlap={overlap}, new_overlap={new_overlap}')


        #### Go to nambu basis:
        E_vec1 = nambu_basis_slow(E_vec1, N_min, N_max)
        E_vec2 = nambu_basis_slow(E_vec2, N_min, N_max)
        
        #### Overlap computation
        vec_indices = list(range(N_min))
        
        N_zero = N_max-N_min
        assert N_zero %2 == 0, f'N_zero:{N_zero} must be even'
        N_zero2 = int(N_zero/2)
        X = zeros([N_zero2,N_zero2], dtype = complex)
        
        for n in range(N_min, N_min + N_zero2):
            list1 = vec_indices + [n]
            E1 = E_vec1[:,list1]
            for m in range(N_min, N_min + N_zero2):
                list2 = vec_indices + [m]
                E2 = E_vec2[:,list2]
                
                B = dagger(E1) @ E2
                X[n-N_min,m-N_min] = det(B)

        return X

# %% function: Compute single braiding step
def compute_braiding_single_step(msys, anyon_loop_indices, path, path_anyon_dict, i_path, ind_step, N_steps, atol=1e-8, check=False, plot_anyons=False):

    """
    Computes the overlap matrix for a single braiding step of the system.   
    The function assumes that the system has periodic boundary conditions (PBC).
    The function works by adjusting the Hamiltonian parameters for the previous anyon movements along the path,
    finding the anyonic eigenstates before and after the braiding step, and computing the inner product of the two sets of eigenstates.
    Parameters
    ----------
    msys : object
        The system object containing the Hamiltonian and other parameters.
    anyon_loop_indices : list       
        List of indices of the anyon loops where the local anyonic modes are to be found.
    path : list 
        List of indices of the bonds along which the anyons are braided.
    path_anyon_dict : dict
        Dictionary containing the mapping of anyon loop indices for each bond in the path.
    i_path : int
        The index of the bond in the path along which the anyon is braided.
    ind_step : int
        The index of the braiding step.
    N_steps : int
        The total number of braiding steps.
    atol : float, optional
        The absolute tolerance for determining the number of zero modes. The default is 1e-8.
    check : bool, optional
        If True, the function will perform additional checks and plot the anyonic modes. The default is False.
    plot_anyons : bool, optional
        If True, the function will plot the anyonic modes before and after the braiding step. The default is False.
    Returns
    -------
    X : np.array
        The overlap matrix for the single braiding step.
    Raises
    ------
    AssertionError
        If the system does not have periodic boundary conditions (PBC). Please use function: "compute_braiding_single_step_obc" for OBC systems.
    AssertionError
        If the transformed vectors after gram-schmidtt are not orthonormal: dagger(T) @ T != np.eye({N_max-N_min}), \n{dagger(E_vec1) @ E_vec1}, \n{dagger(E_vec2) @ E_vec2}    
    """

    #### Adjust for hamiltonian parameters for previous anyon movements along the path
    msys = deepcopy(msys) # do not alter msys during the process
    hop_dict = msys.hop_dict
    hop_index_dict = {hop_dict[key]['index']: key for key in hop_dict.keys()}
    N_path = len(path)
    msys.update = True
    N_sites = msys.N_sites
    anyon_loop_indices = deepcopy(anyon_loop_indices)
    # print(f'before prep loop_indices:{anyon_loop_indices}')

    for j in range(i_path):
        index = path[j]
        edge = hop_index_dict[index]
        Jb = hop_dict[edge]['strength']
        hop_dict[edge]['strength'] = -Jb
        
        loop_index_out, loop_index_in = path_anyon_dict[j]
        
        index_out = anyon_loop_indices.index(loop_index_out)
        anyon_loop_indices[index_out] = loop_index_in

    # print(f'after prep loop indices:{anyon_loop_indices}')


    bond_index = path[i_path]
    edge = hop_index_dict[bond_index] 
    J = hop_dict[edge]['strength']
    
    #### Current hamitlonian parameters
    J_prev = J*(1-2*(ind_step-1)/N_steps)
    hop_dict[edge]['strength'] = J_prev
    msys.hop_dict = hop_dict
    # print(f'J={J}, J_prev = {J_prev}, hop_dict[edge]["strength"]={hop_dict[edge]["strength"]}')
    

    #### Get current anyonic eigenstates
    E_val_prev, E_vec_prev = eigh(msys.ham.toarray())
    E_vec_prev = local_anyon_basis_pbc(msys, anyon_loop_indices=anyon_loop_indices)
    atol = 1e-12
    N_max = sum(E_val_prev < atol)
    N_min = sum(E_val_prev < -atol) 
    

    if plot_anyons== True:
        #### Plot anyons
        fig, ax = plt.subplots(6,N_max-N_min,figsize=[5*(N_max-N_min),30], constrained_layout=True)
        for i in range(N_min, N_max):
            print(f'i = {i}, E={E_val_prev[i]}')
            state = E_vec_prev[:,i]

            state_r = real(state)
            state_i = imag(state)
            state_i[state_i < 1e-14] = 0
            state_phi = angle(state)/pi
            # state_phi[np.abs(state)<1e-14] = 0
            
            
            plot_state(state, msys, fig, ax[0,i-N_min])
            plot_state(state_r, msys, fig, ax[1,i-N_min])
            ax[2,i-N_min].plot(range(len(state_r)), state_r, '-or')
            plot_state(state_i, msys, fig, ax[3,i-N_min])
            print(f'Max imaginary val of state:{max(imag(state))}')
            ax[4,i-N_min].plot(range(len(state_i)), state_i, '-og')
            ax[5,i-N_min].plot(range(len(state_phi)), state_phi, '-ob')
            
        plt.show()

    #### check orthonormality
    trans_vectors = E_vec_prev[:, N_min:N_max]
    assert allclose(dagger(trans_vectors) @ trans_vectors, eye(N_max-N_min)), f'The transformed vectors after gram-shmidtt are not orthonormal: dagger(T) @ T != np.eye({N_max-N_min}), \n{dagger(trans_vectors) @ trans_vectors}'

    #### Move anyons: Hamiltonian parameters for next step
    J_new = J*(1-2*(ind_step)/N_steps)
    hop_dict[edge]['strength'] = J_new
    msys.hop_dict = hop_dict
    # print(f'J={J}, J_prev = {J_new}, hop_dict[edge]["strength"]={hop_dict[edge]["strength"]}\n')
    h = msys.ham


    #### Get new anyonic states
    E_val_new, E_vec_new = eigh(msys.ham.toarray())
    E_vec_new = local_anyon_basis_pbc(msys, anyon_loop_indices=anyon_loop_indices)
    atol = 1e-12
    N_max = sum(E_val_new < atol)
    N_min = sum(E_val_new < -atol)  

    if plot_anyons == True:
        #### Plot anyons
        fig, ax = plt.subplots(6,N_max-N_min,figsize=[5*(N_max-N_min),30], constrained_layout=True)
        for i in range(N_min, N_max):
            print(f'i = {i}, E={E_val_new[i]}')
            state = E_vec_new[:,i]

            state_r = real(state)
            state_i = imag(state)
            state_i[abs(state_i) < 1e-14] = 0
            state_phi = angle(state)/pi
            state_phi[abs(state)<1e-14] = 0

            
            plot_state(state, msys, fig, ax[0,i-N_min])
            plot_state(state_r, msys, fig, ax[1,i-N_min])
            ax[2,i-N_min].plot(range(len(state_r)), state_r, '-or')
            plot_state(state_i, msys, fig, ax[3,i-N_min])
            print(f'Max imaginary val of state:{max(imag(state))}')
            ax[4,i-N_min].plot(range(len(state_i)), state_i, '-og')
            ax[5,i-N_min].plot(range(len(state_phi)), state_phi, '-ob')
            
        plt.show()

    #### check orthonormality
    trans_vectors = E_vec_new[:, N_min:N_max]
    assert allclose(dagger(trans_vectors) @ trans_vectors, eye(N_max-N_min)), f'Found in testing. NOT FLAGGED IN THE ORIGINAL FUNCTION. The transformed vectors after gram-shmidtt are not orthonormal: dagger(T) @ T != np.eye({N_max-N_min}), \n{dagger(trans_vectors) @ trans_vectors}'




    #### Compute inner product
    if msys.syst_params['pbc'] == False:
        overlap_f = many_body_product_state_overlap_fermionic(E_vec_prev, E_vec_new, N_min=N_min, N_max=N_max -1) 
    else:
        overlap_f = many_body_product_state_overlap_fermionic(E_vec_prev, E_vec_new, N_min=N_min, N_max=N_max) 

    return overlap_f

