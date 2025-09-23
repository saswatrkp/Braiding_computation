# %% import modules
from numpy import conj, transpose, zeros, sqrt, vdot, allclose, arange, abs, log, ones
from numpy.linalg import qr, eigvalsh
from copy import deepcopy
from kwant import plot as kplot



# %% support functions
# %% Supporting functions

def switch_grid(ax):
    ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


def turnoff_labels(ax):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)



def dagger(H):
    """Returns the hermitian conjugate of the operator

    Parameters
    ----------
    H : 2D array
        Matrix representation of the operator in some basis

    Returns
    -------
    H_dag: 2D array
        Hermitian conjugate of H
    """
    H_dag = transpose(conj(H))
    return H_dag





def Pos_mat(syst):
    """
    Creates position matrix which stores the index of the sites according to their positions.

    Parameters
    ----------
    syst : kwant.Builder.finalized()
        Flinalized Builder object for SG-3.

    Returns
    -------
    st_pos : np.array
        Array with st_pos[i,:] = [site index, x, y].

    """
    st_pos= zeros((syst.graph.num_nodes,3))
    for i in range(syst.graph.num_nodes):
        st_pos[i,0]=i
        st_pos[i,1]=syst.pos(i)[0]
        st_pos[i,2]=syst.pos(i)[1]
    return st_pos





def norm(U):
    """
    Calculates norm of a vector

    Parameters
    ----------
    U : np.array
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return sqrt(vdot(U,U))





def dot(U,V):
    """
    Calculates inner product of two vectors, taking the complex conjugate of the first 
    argument.

    Parameters
    ----------
    U : np.array
        DESCRIPTION.
    
    V : np.array
        DESCRIPTION.
    

    Returns
    -------
    None.

    """
    return vdot(U,V)





def gram_schmidt(A, target):
    """
    Orthonormalizes the set of vectors in X with respect to a target vector.

    Parameters
    ----------
    A : np.array
        Array of linearly independent column vectors {x1,x2,...xn} such that
        xi = X[:,i]
        
    target : np.array
        Target vector

    Returns
    -------
    Q : np.array
        Orthonormalized vectors

    """
    X = deepcopy(A)
    N = X.shape[1]
    Ns = X.shape[0]
    assert X.ndim==2, f'Enter a 2D-array. Current array dimension:{len(X.shape)}'
    dot_list = [abs(dot(target, X[:,i])) for i in range(X.shape[1])]
    j = dot_list.index(max(dot_list))
    X[:,j] = deepcopy(X[:,0])
    X[:,0] = target
    Q, R = qr(X)
    return Q




def Bi_partition_Hamiltonian(h, A_indices, B_indices, check=False):
    """
    Convert the hamitlonian H into H =  [[H_A, H_AB],[H_BA H_B]] when 
    the system is bi-partitioned in to A and B.

    Parameters
    ----------
    h : sp.sparse_array
        Hamiltonian of the system in local basis.
    A_indices : list
        Indices of sites in subsystem A.
    B_indices : list
        Indices of sites in subsystem B.

    Returns
    -------
    H : np.array
        Final form of the Hamiltonian.

    """
    
    ham = h.toarray()
    N_sites = ham.shape[0]
    NA = len(A_indices)
    NB = len(B_indices)
    assert NA+NB==N_sites, f'NA:{NA} + NB:{NB} != N_sites:{N_sites}'
    
    #### Check that A + B = 1
    assert allclose(arange(0,N_sites),list(set(A_indices).union(set(B_indices)))), 'A + B != 1.'
    assert len(list(set(A_indices).intersection(set(B_indices))))==0, 'Common elements in A and B.'
    
    #### Create Permutation
    col_ind = A_indices + B_indices
    row_ind = arange(0,N_sites)
    data = ones([N_sites])
    from scipy.sparse import csr_array
    perm = csr_array((data, (row_ind, col_ind)), shape=(N_sites, N_sites))
    P = perm.toarray()
    P_inv = transpose(perm)
    
    ham_new = P @ ham @ P_inv
    
    if check ==True:
        print(f'perm: {allclose(P @ row_ind, col_ind)}')
        print(f'inverse_perm: {allclose(ham, P_inv @ ham_new @ P)}')
        E_val = eigvalsh(ham)
        E_val_p = eigvalsh(ham_new)
        print(f'E-val_check: {allclose(E_val,E_val_p)}')

    return ham_new, P




def plot_state(vec, msys, fig, ax, cmap = 'gist_heat_r', scale='log'):
    """Plots the state of a system using kwant's plotter function.

    Parameters
    ----------
    vec : 1d np.array or a list of numbers
        State of a system
    msys : model class for the system
        Representation of the system. msys.syst must be a kwant builder
    fig : matplotlib figure
        Figure to plot on 
    ax : matplotlib axis
        Axis on which the plot is to be plotted
    """
    if max(abs(vec)) < 1e-13:
        state_color = zeros(len(vec))
    else:
        state_color = abs(vec)/(norm(abs(vec)))

    state_color[state_color<1e-12] = 1e-16
    if scale=='log':
        state_color = log(state_color)
    elif scale=='linear':
        pass
    else:
        raise ValueError("scale must be either 'log' or 'linear'")
    
    kplot(msys.syst, site_color=state_color, cmap = cmap, ax=ax)
    cb = fig.colorbar(mappable=ax.collections[0], ax=ax)
    ax.collections[0].set_clim(-8,0)
    # from matplotlib.colors import Normalize
    # color_norm = Normalize(vmin=-8, vmax=0)
    # ax.collections[0].set_norm(color_norm)
    if scale=='log':
        cb.ax.set_title(r'$\log(\rho)$', fontsize=20)
    elif scale=='linear':
        cb.ax.set_title(r'$\rho$', fontsize=20)
    else:
        raise ValueError("scale must be either 'log' or 'linear'")
    turnoff_labels(ax)



def create_anyons(msys, anyon_loop_indices):
    """
    Creates anyons in the system.

    Parameters
    ----------
    msys : model class for the system
        Representation of the system.
    anyon_loop_indices : list
        List of loop indices where anyons are to be created.
    """
    N_anyons = len(anyon_loop_indices)
    assert N_anyons % 2 == 0, f'# of anyons must be even. Current list contains #{N_anyons} anyons.'
    msys.update = True
    for i in range(int(N_anyons/2)):
        msys.anyons_pair(anyon_loop_indices[i], anyon_loop_indices[i+int(N_anyons/2)])
    msys.update = False



def path_anyon_dict_2_path(msys, path_anyon_dict):
    """
    Converts a dictionary of anyon paths to a list of path indices.

    Parameters
    ----------
    msys : model class for the system
        Representation of the system.
    path_anyon_dict : dict
        Dictionary of anyon paths.

    Returns
    -------
    path : list
        List of path indices.
    """
    loop_dict = msys.loop_dict
    hop_dict = msys.hop_dict
    path = []
    for indices in path_anyon_dict.values():
        assert len(indices) == 2, f'Exactly 2 anyons share a bond. Present number of anyon indices:{len(indices)}'
        edge_set_0 = set([data['hop_sites'] for data in loop_dict[indices[0]]['edges']])
        edge_set_1 = set([data['hop_sites'] for data in loop_dict[indices[1]]['edges']])
        bond_sites = list(edge_set_0 & edge_set_1)
        assert len(bond_sites)==1, f'There can be only one common bond between two loops. Present number of common bonds={len(bond_sites)}'
        bond_sites = bond_sites[0]
        bond_index = hop_dict[bond_sites]['index']
        path.append(bond_index)
    return path




def sel_sites(msys, loop_index, eps = 1e-5):
    """
    Select sites around the holes of SG-3 for full chern number computation.

    Parameters
    ----------
    msys: model class of Kitaev SG-3
        system for Kitaev SG-3
    x0 : float
        x-coordinate of the cross-hair
    y0 : float
        y-coordinate of the cross-hair

    Returns
    -------
    new_site_indices : np.array
        Array of indices of the selected sites.
    rest_site_indices : np.array
        Array of indices of the unselected sites.

    """
    
    #### Define system
    syst = msys.syst
    id_by_site = {site: i for i, site in enumerate(syst.sites)}

    #### Find sites encircling the loop using loop dict    
    loop_edges = msys.loop_dict[loop_index]['edges']
    site_list = []
    for edge in loop_edges:
        sites = edge['hop_sites']
        site_list.extend(sites)
    site_list = list(set(site_list))
        
    
    new_site_indices=[]
    for site in site_list:
        new_site_indices.append(id_by_site[site])
    
        
    rest_site_indices=list(set(list(range(syst.graph.num_nodes)))-set(new_site_indices))
    return new_site_indices, rest_site_indices




def anyon_position(msys, loop_index):
    """Gives the position of the anyon given that the anyon is localized on a given loop

    Parameters
    ----------
    msys : model Kitaev_SG3
        System for the Kitaev model on SG-3
    loop_index : int
        index of the loop on which the anyon is localized

    Returns
    -------
    tuple
        Positoins of the centroid of the loop as the effective location of the anyon.
    """
    edges = msys.loop_dict[loop_index]['edges']
    N_L = len(edges)
    X = 0
    Y = 0
    for ed in edges:
        site1, site2 = ed['hop_sites']
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        X = X + (x1 + x2)/2
        Y = Y + (y1 + y2)/2

    return X/N_L, Y/N_L


def plot_system(model, anyon_loop_indices, ax, show_bond_indices=False, show_loop_indices=False, kwant_style=False, **kwargs):
    """
    Plots the system using kwant's plotter function.

    Parameters
    ----------
    model : model class for the system
        Representation of the system. msys.syst must be a kwant builder
    anyon_loop_indices : list
        List of loop indices where anyons are to be created
    ax : matplotlib axis
        Axis on which the plot is to be plotted
    show_bond_indices : bool, optional
        Whether to show bond indices. Defaults to False
    show_loop_indices : bool, optional
        Whether to show loop indices. Defaults to False
    kwant_style : bool, optional
        Whether to use kwant style plotting. Defaults to False
    **kwargs : dict
        Additional keyword arguments to pass to model.show_system
    """
    create_anyons(msys = model, anyon_loop_indices=anyon_loop_indices)
    model.show_system(
        show_bond_indices=show_bond_indices,
        show_loop_indices=show_loop_indices,
        kwant_style=kwant_style,
        ax=ax,
        **kwargs
    )
    turnoff_labels(ax)
    switch_grid(ax)