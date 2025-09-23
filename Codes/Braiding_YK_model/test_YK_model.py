# %% import modules
import numpy as np
import matplotlib.pyplot as plt
from time import time


import sys
import os
# Add the parent directory to Python path to find the modules folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import Yao_Kivelson
from modules.support_functions import create_anyons, plot_system, path_anyon_dict_2_path
from modules.braiding_functions import local_anyon_basis_pbc, compute_braiding_single_step
from Braiding_YK_model.plotting_functions import plot_energy_spectrum, plot_zero_modes, plot_local_anyon_basis


# %% Define parameters

#### Define the system parameters
g = 4
alpha = 0.25*np.pi
G = g
L = 8
pbc = False
add_b_site = False
W = 0
u = 0
syst_params = {'G':G, 'L': L, 'W':W, 'u':u, 'pbc':pbc, 'add_b_site':add_b_site}
t = np.abs(np.sin(alpha))
J = np.abs(np.cos(alpha))
hop_params = {'ty':1j*t, 'tx':1j*t, 'tz':1j*t, 'Jy':1j*J, 'Jx':1j*J, 'Jz':1j*J}



#### Define the anyon parameters

anyon_loop_indices = [100,137,93,147]
# anyon_loop_indices = [100,59,90,137]




#### Define Braiding path parameters

## path 1: Double exchange of anyons in interesecting path in PBC
path_anyon_dict = {0:[93,35], 1:[35,120], 2:[120,59], 3:[147,78], 4:[78,35], 5:[35,153], 6:[153,90], 7:[59,120], 8:[120,35], 9:[35,93], 10:[90,153], 11:[153,35], 12:[35,78], 13:[78,147]}
N_path = len(path_anyon_dict)




# %% Initialize the model

model = Yao_Kivelson(syst_params=syst_params, hop_params = hop_params)
print('Number of sites:', model.N_sites)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
plot_system(model, anyon_loop_indices, ax, show_loop_indices=True, kwant_style=False, show_bond_indices=True)
plt.show()




# %% Plot the energy spectrum
    
fig, ax = plt.subplots(constrained_layout=True)
plot_energy_spectrum(syst_params=syst_params, hop_params=hop_params, anyon_loop_indices=anyon_loop_indices, ax=ax)
plt.show()




# %% Plot the zero modes

plot_zero_modes(syst_params=syst_params, hop_params=hop_params, anyon_loop_indices=anyon_loop_indices, pbc=True)





# %% Plot the local anyonic basis in PBC system

plot_local_anyon_basis(syst_params=syst_params, hop_params=hop_params, anyon_loop_indices=anyon_loop_indices)




# %% function: Test single braiding step

def test_single_braiding_step(path_anyon_dict, ind_step, N_steps, i_path, check=False, plot_anyons=False):
    """
    Tests a single braiding step of the system.

    Parameters
    ----------
    model : model class for the system
        Representation of the system. msys.syst must be a kwant builder
    
    """
    syst_params['pbc'] = True
    model = Yao_Kivelson(syst_params=syst_params, hop_params=hop_params)
    create_anyons(msys=model, anyon_loop_indices=anyon_loop_indices)
    print('Number of sites:', model.N_sites)

    T_evec = local_anyon_basis_pbc(msys=model, anyon_loop_indices=anyon_loop_indices, check=False)

    path = path_anyon_dict_2_path(msys=model, path_anyon_dict=path_anyon_dict)
    N_path = len(path)
    print(f'Braiding path: {path}')
    print(f'Number of steps in the path: {N_path}')

    print(f'i_path={i_path}, ind_step={ind_step}, N_steps = {N_steps}')
    X = compute_braiding_single_step(msys=model, anyon_loop_indices=anyon_loop_indices, path=path, path_anyon_dict=path_anyon_dict, i_path=i_path, ind_step=ind_step,N_steps= N_steps, check=check, plot_anyons=plot_anyons)

    return X

ind_step = 0 # np.random.randint(low=0, high=50)
N_steps = 20
i_path = 5
overlap = test_single_braiding_step(path_anyon_dict=path_anyon_dict, ind_step=ind_step, N_steps=N_steps, i_path=i_path, check=True)
print(f'Norm of the overlap matrix: {np.linalg.norm(overlap)}')
    

# %% Loop over all ind_steps:
N_steps = 50
t0 = time()
norm_overlap_list = []
overlap_list = []
for i_path in range(len(path_anyon_dict)):
    for ind_step in range(1, N_steps+1):
        overlap = test_single_braiding_step(path_anyon_dict=path_anyon_dict, ind_step=ind_step, N_steps=N_steps, i_path=i_path, check=False)
        print(f'Norm of the overlap matrix: {np.linalg.norm(overlap)}')
        overlap_list.append(overlap)
        norm_overlap_list.append(np.linalg.norm(overlap))
t1 = time()
print(f'\nTime taken for {N_steps*len(path_anyon_dict)} steps: {t1-t0} seconds\n')


# %% Plot the norm overlap
plt.plot(range(N_path*N_steps), norm_overlap_list, '-o')
plt.show()

# %% Print the total braiding matrix
from functools import reduce
from numpy import matmul
total_flux = reduce(matmul, overlap_list)
print(f'Total flux matrix:\n Absolute:{np.abs(total_flux)}\n Angle:{np.angle(total_flux)/np.pi} \n')

# %%
