# %% import modules
import sys
import os
# Add the parent directory to Python path to find the modules folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import Yao_Kivelson
from modules.support_functions import create_anyons, switch_grid, plot_state, plot_system
from modules.braiding_functions import local_anyon_basis_pbc


import matplotlib.pyplot as plt
from numpy import arange, linspace
from numpy.linalg import eigvalsh, eigh

# %% function: plot energy spectrum
def plot_energy_spectrum(syst_params, hop_params, anyon_loop_indices, ax):
    """
    Plots the energy spectrum of the system.
    Parameters
    ----------
    syst_params : dict
        Dictionary containing the system parameters.
    hop_params : dict
        Dictionary containing the hopping parameters.
    anyon_loop_indices : list
        List of lists containing the indices of the anyon loops.
    ax : matplotlib.axes.Axes
        Axes object to plot the energy spectrum on. The default is None.
    Returns
    -------
    None.
        
    """

    markersize = 10
    inset_markersize = 2
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1)


    #### Generate model in OBC
    syst_params['pbc'] = False
    model_obc = Yao_Kivelson(syst_params=syst_params, hop_params=hop_params)
    H_obc = model_obc.ham
    E = eigvalsh(H_obc.toarray()) ## without anyons
    x = 1/len(E) * arange(0, len(E))
    ax.plot(x, E, 'og', markersize=markersize, mfc = 'none',label='OBC')
    inset_ax.plot(x, E, 'og', markersize=inset_markersize, mfc = 'none',label='OBC')
    create_anyons(msys=model_obc, anyon_loop_indices=anyon_loop_indices) ## create anyons
    H_obc = model_obc.ham
    E = eigvalsh(H_obc.toarray()) ## with anyons
    x = 1/len(E) * arange(0, len(E))
    ax.plot(x, E, '*g', markersize=markersize, mfc = 'none', label='OBC_anyons')
    inset_ax.plot(x, E, '*g', markersize=inset_markersize, mfc = 'none', label='OBC_anyons')


     #### Generate model in PBC

    syst_params['pbc'] = True
    model_pbc = Yao_Kivelson(syst_params=syst_params, hop_params=hop_params)
    H_pbc = model_pbc.ham
    E = eigvalsh(H_pbc.toarray()) ## without anyons
    x = 1/len(E) * arange(0, len(E))
    ax.plot(x, E, 'or', markersize=markersize, mfc = 'none', label='PBC')
    inset_ax.plot(x, E, 'or', markersize=inset_markersize, mfc='none', label='PBC')
    create_anyons(msys=model_pbc, anyon_loop_indices=anyon_loop_indices) ## create anyons
    H_pbc = model_pbc.ham
    E = eigvalsh(H_pbc.toarray()) ## with anyons
    x = 1/len(E) * arange(0, len(E))
    ax.plot(x, E, '*r', markersize=markersize, mfc = 'none', label='PBC_anyons')
    inset_ax.plot(x, E, '*r', markersize=inset_markersize, mfc = 'none', label='PBC_anyons')
    
    
    ax.legend(fontsize=12)    
    switch_grid(ax)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(r'Index $(\frac{n}{N})$', fontsize=20)
    ax.set_ylabel('Energy', fontsize=20)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlim(0.475, 0.525)


    
    inset_ax.set_xlabel(r'Index $(\frac{n}{N})$', fontsize=10)
    inset_ax.set_ylabel('Energy', fontsize=10)
    inset_ax.axes.get_xaxis().set_visible(False)
    inset_ax.yaxis.set_tick_params(labelsize=8)
    inset_ax.set_title("Full Spectrum", fontsize=10)
    

# %% function: plot zero modes

def plot_zero_modes(syst_params, hop_params, anyon_loop_indices, pbc):
    """
    Plots the zero modes of the system.

    Parameters
    ----------
    syst_params : dict
        Dictionary containing the system parameters.
    hop_params : dict
        Dictionary containing the hopping parameters.
    anyon_loop_indices : list
        List of lists containing the indices of the anyon loops.
    pbc : bool, optional
        Whether to use periodic boundary conditions. The default is True.
    Returns
    -------
    None.
    """
    if pbc == True:
        syst_params['pbc'] = True
    else:
        syst_params['pbc'] = False
    model = Yao_Kivelson(syst_params=syst_params, hop_params=hop_params)
    create_anyons(msys=model, anyon_loop_indices=anyon_loop_indices)
    model.show_system()
    plt.show()
    H = model.ham.toarray()
    E_val, E_vec = eigh(H)
    

    
    N_max = sum(E_val < 1e-10)
    N_min = sum(E_val < -1e-10)
    N_zero = N_max - N_min
    print(f'Number of zero modes: {N_max, N_min, N_zero}')

    fig, ax = plt.subplots(1,N_zero,figsize=[5*(N_zero),5], constrained_layout=True)
    for i in range(N_min, N_max):
        state = E_vec[:,i]
        plot_state(state, model, fig, ax[i-N_min])  
    plt.show()




# %% function: plot local anyonic basis in PBC system
def plot_local_anyon_basis(syst_params, hop_params, anyon_loop_indices, pbc=True, check=True):
    """
    Plots the local anyonic basis of the system in PBC.
    Parameters
    ----------
    syst_params : dict
        Dictionary containing the system parameters.
    hop_params : dict
        Dictionary containing the hopping parameters.
    anyon_loop_indices : list
        List of lists containing the indices of the anyon loops.
    pbc : bool, optional
        Whether to use periodic boundary conditions. The default is True.
    Returns
    -------
    None.
    """
    syst_params['pbc'] = True
    model = Yao_Kivelson(syst_params=syst_params, hop_params=hop_params)
    create_anyons(msys=model, anyon_loop_indices=anyon_loop_indices)
    model.show_system()
    plt.show()

    T_evec = local_anyon_basis_pbc(msys=model, anyon_loop_indices=anyon_loop_indices, check=check)

    H = model.ham.toarray()
    E_val = eigvalsh(H)
    
    N_max = sum(E_val < 1e-10)
    N_min = sum(E_val < -1e-10)
    N_zero = N_max - N_min
    print(f'Number of zero modes: {N_max, N_min, N_zero}')
    
    fig, ax = plt.subplots(1,N_zero,figsize=[5*(N_zero),5], constrained_layout=True)
    for i in range(N_min, N_max):
        state = T_evec[:,i]
        plot_state(state, model, fig, ax[i-N_min])  
    plt.show()