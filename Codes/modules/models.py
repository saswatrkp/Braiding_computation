#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:11:03 2023

@author: Saswat Sarangi
"""

# %% Import modules
from numpy import array, zeros, nonzero, log, sin, cos, sqrt, pi, inf, nan, heaviside, sign, exp
from numpy.random import random
import kwant
from copy import deepcopy
import matplotlib.pyplot as plt

# %% Parent model class
class __model_system__:
    
    def __init__(self, m_syst_params, m_hop_params, m_sites, m_hop_dict, m_loop_dict):
        self.update = False
        self.__assign_properties__(m_syst_params, m_hop_params, m_sites, m_hop_dict, m_loop_dict)
        self.init_params = deepcopy(self.__params)
        self.__init_loop_dict = deepcopy(self.__loop_dict)
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params=self.__params, sparse='True')
        self.vortices = []
        
    
    def __assign_properties__(self, m_syst_params, m_hop_params, m_sites, m_hop_dict, m_loop_dict):
         self.syst_params = m_syst_params
         self.hop_params = m_hop_params
         self.__hop_dict = m_hop_dict
         self.sites = m_sites
         self.__loop_dict = m_loop_dict
         self.__w_dict = self.__w_jk_dict__()
         self.__params = dict(hop_dict = self.__hop_dict, w_dict = self.__w_dict)
        
        
        
    @property
    def hop_dict(self):
        return self.__hop_dict
    
    @hop_dict.setter
    def hop_dict(self, new_hop_dict):
        self.__hop_dict = new_hop_dict
        self.__params['hop_dict'] = self.__hop_dict
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params=self.__params, sparse='True')
    
    @property
    def w_dict(self):
        return self.__w_dict
    
    @w_dict.setter
    def w_dict(self, new_w_dict):
        self.__w_dict = new_w_dict
        self.__params['w_dict'] = self.__w_dict
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params = self.__params, sparse='True')
    
    @property
    def loop_dict(self):
        return self.__loop_dict
    
    @loop_dict.setter
    def loop_dict(self, new_loop_dict):
        self.__loop_dict = new_loop_dict
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params = self.__params, sparse='True')
    
    
    
    def __w_jk_dict__(self):
        """
        Create dictionary for adding bond disorder in the Kitaev model.

        Parameters
        ----------
        w_list : list 
            list of Disorder values.
        hops_dict : dict of dict
            dictionary of dictionaries corresponding to all hopping elements in SG-3.

        Returns
        -------
        w_dict : dict
            Dictionary of disorder values.
            Format: {'hop_sites': flux_value}

        """
        W = self.syst_params['W']
        w_list = random([self.N_bonds]) * W - W/2
        assert len(w_list) == self.N_bonds, f'len(w_list):{len(w_list)} does not match N_bonds:{self.N_bonds}'
        w_dict = {}
        for ibond,hop in enumerate(self.__hop_dict.keys()):
            w_dict[hop] = w_list[ibond]
        return w_dict
    
    
    
    def __build_system__(self):
        """
        Builds the kwant system from the hopping dictionaries.

        """
        self.syst_prim = kwant.Builder()
        
        for site in self.sites:
            self.syst_prim[site] = self.syst_params['u']
        for hop in self.hop_dict.keys():
            self.syst_prim[hop] = self.__hop_strength__
               
        self.syst = self.syst_prim.finalized()
    
    
    
    
    def show_system(self, ax=None, show_bond_indices = False, show_loop_indices=False, kwant_style = False, show_u_indices = False, transperency_sites = 1, transperency_bonds = 1):
        """
        Plots the system with sites and bonds. Shows the indices of the Loops.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axis for plotting the system. Creates a new axis when ax==None.
            The default is None.
        show_bond_indices : bool, optional
            Determines if bonds indices are shown.
            The default is False.
        show_bond_indices : bool, optional
            Determines if bonds indices are shown.
            The default is False.

        Returns
        -------
        None.

        """
        if show_u_indices==True:
            show_bond_indices=False
        
        def plot_sys(axs, transperency_sites, transperency_bonds):
            hop_dict = self.hop_dict
            hop_keys = list(hop_dict.keys())
            for key in hop_keys:
                site1, site2 = key
                x1,y1 = site1.pos
                x2,y2 = site2.pos
                line_color = self.__hop_color__(site1, site2)
                line_style = self.__hop_ls__(site1, site2)
                line_width = 30 * self.__hop_lw__(site1, site2)
                axs.plot([x1,x2], [y1,y2], ls = line_style, color = line_color, lw = line_width, alpha = transperency_bonds)
                if show_bond_indices==True:
                    text = hop_dict[key]['index']
                    axs.text((x1+x2)/2, (y1+y2)/2, text, fontsize = 8)  
                elif show_u_indices==True:
                    text = hop_dict[key]['u_index']
                    axs.text((x1+x2)/2, (y1+y2)/2, text, fontsize = 8)  
                    
            
            x=[site.pos[0] for site in self.sites]
            y=[site.pos[1] for site in self.sites]
            axs.plot(x,y, 'o', markerfacecolor = 'tab:blue', alpha = transperency_sites)
        
        def show_loops(axs):
            for i in range(self.N_loops):
                w_L = self.loop_dict[i]['w_L']
                edges = self.loop_dict[i]['edges']
                N_L = len(edges)
                X = 0
                Y = 0
                for ed in edges:
                    site1, site2 = ed['hop_sites']
                    x1,y1 = site1.pos
                    x2,y2 = site2.pos
                    X = X + (x1+x2)/2
                    Y = Y + (y1+y2)/2
                axs.text(X/N_L, Y/N_L, i, color = 'r', fontsize = 8)
                if w_L == -1:
                    axs.scatter(X/N_L, Y/N_L, c='r', alpha = 0.5, s = 400, marker = '*')

        def show_anyons(axs):
            for i in range(self.N_loops):
                w_L = self.loop_dict[i]['w_L']
                edges = self.loop_dict[i]['edges']
                N_L = len(edges)
                X = 0
                Y = 0
                for ed in edges:
                    site1, site2 = ed['hop_sites']
                    x1,y1 = site1.pos
                    x2,y2 = site2.pos
                    X = X + (x1+x2)/2
                    Y = Y + (y1+y2)/2
                if w_L == -1:
                    axs.scatter(X/N_L, Y/N_L, c='r', alpha = 0.5, s = 400, marker='*')

            
        if ax==None:
            fig, ax = plt.subplots(constrained_layout=True)
            if kwant_style==True:
                kwant.plot(self.syst_prim, hop_color=self.__hop_color__, hop_lw = self.__hop_lw__, ax=ax)
                show_anyons(ax)
            else:
                plot_sys(ax, transperency_sites, transperency_bonds)
                show_anyons(ax)
            if show_loop_indices==True:
                show_loops(ax)
            
        else:
            if kwant_style==True:
                kwant.plot(self.syst_prim, hop_color=self.__hop_color__, hop_lw = self.__hop_lw__, ax=ax)
                show_anyons(ax)
            else:
                plot_sys(ax, transperency_sites, transperency_bonds)
                show_anyons(ax)
            if show_loop_indices==True:
                show_loops(ax)
    
    
    
    
    
    def __hop_lw__(self, site1, site2):
        hop_type = self.hop_dict[(site1,site2)]['type']
        if hop_type == 'ty' or hop_type == 'tx' or hop_type == 'tz':
            return 0.09
        if hop_type == 'Jy' or hop_type == 'Jx' or hop_type == 'Jz':
            return 0.1
        else:
            return 0    
    
    def __hop_ls__(self,site1, site2):
        hop_type = self.hop_dict[(site1,site2)]['type']
        if hop_type == 'ty' or hop_type == 'tx' or hop_type == 'tz':
            return (0, (0.5,0.5))
        if hop_type == 'Jy' or hop_type == 'Jx' or hop_type == 'Jz':
            return '-'
        else:
            return 0
        
    def __hop_color__(self,site1, site2):
        hop_type = self.hop_dict[(site1,site2)]['type']
        if hop_type == 'ty' or hop_type == 'Jy':
            return 'green'
        if hop_type == 'tx' or hop_type == 'Jx':
            return 'red'
        if hop_type == 'tz' or hop_type == 'Jz':
            return 'blue'
        else:
            return 'black'
        
                          
    def __hop_strength__(self,site1, site2):
        if self.hop_dict[(site1,site2)]['type'] == 'tz' or self.hop_dict[(site1,site2)]['type'] == 'Jz':
            u_jk = self.hop_dict[(site1, site2)]['u_jk']
            return u_jk * self.hop_dict[(site1,site2)]['strength']  + 1j*self.w_dict[(site1,site2)]
        else:
            return self.hop_dict[(site1,site2)]['strength'] + 1j*self.w_dict[(site1,site2)]
        
        
    def __vortices__(self):
        loop_dict = self.__loop_dict
        N_loops = self.N_loops
        loops=[]
        for i in range(N_loops):
            if loop_dict[i]['w_L'] == -1:
                loops.append(i)
        return loops
        
        
    def anyons_from_u_flips(self, flips):
        if self.update==False:
            create_anyons = __create_anyons__(self.init_params, self.__init_loop_dict)
        else:
            create_anyons = __create_anyons__(self.__params, self.__loop_dict)
        self.__hop_dict, self.__loop_dict = create_anyons.from_u_flips(flips)
        self.__params = dict(hop_dict = self.__hop_dict, w_dict = self.__w_dict)
        self.vortices = self.__vortices__()
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params = self.__params, sparse = 'True')
    
        
    
    def anyons_from_b_flips(self, flips):
        if self.update==False:
            create_anyons = __create_anyons__(self.init_params, self.__init_loop_dict)
        else:
            create_anyons = __create_anyons__(self.__params, self.__loop_dict)
        self.__hop_dict, self.__loop_dict = create_anyons.from_b_flips(flips)
        self.__params = dict(hop_dict = self.__hop_dict, w_dict = self.__w_dict)
        self.vortices = self.__vortices__()
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params = self.__params, sparse = 'True')
        
    
    def anyons_pair(self, l1, l2):
        if self.update==False:
            create_anyons = __create_anyons__(self.init_params, self.__init_loop_dict)
        else:
            create_anyons = __create_anyons__(self.__params, self.__loop_dict)
        self.__hop_dict, self.__loop_dict, N_flips = create_anyons.vortex_pair(l1, l2, self.N_loops)
        self.__params = dict(hop_dict = self.__hop_dict, w_dict = self.__w_dict)
        self.vortices = self.__vortices__()
        self.__build_system__()
        self.ham = self.syst.hamiltonian_submatrix(params = self.__params, sparse = 'True')
        return N_flips

    def n_flips_per_anyon_pair(self, l1, l2):
        if self.update==False:
            create_anyons = __create_anyons__(self.init_params, self.__init_loop_dict)
        else:
            create_anyons = __create_anyons__(self.__params, self.__loop_dict)
            
        return create_anyons.number_of_flips_per_vortex_pair(l1, l2, self.N_loops)


    def __hopping_flux__(self, site1, site2, x0_list, y0_list, phi_list):

        N_tubes = len(phi_list)
        assert len(x0_list)==N_tubes, f'Number of x0:{len(x0_list)} points does not match with number of tubes:{N_tubes}.'
        assert len(y0_list)==N_tubes, f'Number of y0:{len(y0_list)} points does not match with number of tubes:{N_tubes}.'
        
        (x1,y1)=site1.pos
        (x2,y2)=site2.pos
        def flux_new(x1,y1,x2,y2,x0,y0, phi):
            def theta_h(x):
                return heaviside(x,1)
            def slope(x1,y1,x2,y2):
                if y2-y1==0:
                    mx=0
                    my=inf*sign(x2-x1)				
                elif x2-x1==0:
                    mx=inf*sign(y2-y1)
                    my=0
                else:
                    mx=(y2-y1)/(x2-x1)
                    my=1/mx
                return mx,my
            [mx,my]=slope(x1,y1,x2,y2)
            def PHI_X(x1,y1,x2,y2):
                return 2*pi*phi*(theta_h(mx*(x0-x1)-(y0-y1)))*(theta_h(x1-x0)*theta_h(x0-x2)-theta_h(x0-x1)*theta_h(x2-x0))	
            def PHI_Y(x1,y1,x2,y2):
                return -2*pi*phi*(theta_h(my*(y0-y1)-(x0-x1)))*(theta_h(y1-y0)*theta_h(y0-y2)-theta_h(y0-y1)*theta_h(y2-y0))
            return PHI_X(x1,y1,x2,y2)+PHI_Y(x1,y1,x2,y2)
        
        PHI = 0
        for n in range(len(x0_list)):
            x0 = x0_list[n]
            y0 = y0_list[n]
            phi = phi_list[n]
            PHI= PHI + flux_new(x1,y1,x2,y2,x0,y0,phi)
            
        return -1*exp(-0.5j * (PHI))	

    def __hop_add_flux__(self, site1, site2, x0_list, y0_list, phi_list):
        strength = self.__hop_strength__(site1, site2) * self.__hopping_flux__(site1, site2, x0_list, y0_list, phi_list)
        return strength


    def add_flux(self, x0_list, y0_list, phi_list):
        flux_params = dict(hop_dict = self.hop_dict, x0_list=x0_list, y0_list=y0_list, phi_list=phi_list)
        syst_prim_flux = self.syst_prim 

        for hop in self.hop_dict.keys():
            site1, site2 = hop
            syst_prim_flux[site1, site2] = self.__hop_add_flux__
            
        syst_flux = syst_prim_flux.finalized()
        h_flux = syst_flux.hamiltonian_submatrix(params = flux_params, sparse=True)
        return h_flux, syst_flux, syst_prim_flux, flux_params

        
        

    
# %% Create anyons
class __create_anyons__:
    
    def __init__(self, params, loop_dict):
            self.__hop_dict = deepcopy(params['hop_dict'])
            self.__w_dict = deepcopy(params['w_dict'])
            self.__loop_dict = deepcopy(loop_dict)
        
        
        
    
    def from_u_flips(self, flips):
        """
        Creates anyons by flipping the Z_2 gauge degrees of freedom on the z-links of the Kitaev-like model.
        This flips the 'u_jk' in the hop_dict without changing anything else. 
        Also, updates the eigen_values of the loop_operators 'w_L' in the loop_dict.
        
        ASSUMPTION: the single particle model has been obtained using Jordan-Wigner transformation of the 
        spin model, such that the Z_2 gauge degrees of freedom lie on the z-links of the system.

        Parameters
        ----------
        flips : list
            List of the indices of the z-bonds for which the u_jk needs to be flipped.

        Returns
        -------
        hop_dict : dict
            Modified 'hop_dict' after the flip
        
        loop_dict: dict
            Modified 'loop_dict' after the flip
 
        """
        
        loop_dict = self.__loop_dict
        hop_dict = self.__hop_dict
        
        h_index_2_site = {hop_dict[key]['index']:key for key in hop_dict.keys()}
        for flip in flips:
            sites = h_index_2_site[flip]
            if hop_dict[sites]['type'] == 'tz' or hop_dict[sites]['type'] == 'Jz' :
                hop_dict[sites]['u_jk'] = - hop_dict[sites]['u_jk']
                loop_indices = hop_dict[h_index_2_site[flip]]['loop_indices']
                for loop in loop_indices:
                    loop_dict[loop]['w_L'] = -loop_dict[loop]['w_L']
            else:
                print(f'Bond with index:{flip} is not a z-link.')
                
                
        return hop_dict, loop_dict
            
    
    def from_b_flips(self, flips):
        """
        Creates anyons by flipping the sign of the hopping amplitude of the bonds of the Kitaev-like model.
        This flips the sign of the 'hop_strenghth' of bonds in the 'hop_dict'.
        Also, updates the eigen_values of the loop_operators 'w_L' in the loop_dict.
        
        Parameters
        ----------
        flips : list
            List of the indices of the z-bonds for which the u_jk needs to be flipped.

        Returns
        -------
        hop_dict : dict
            Modified 'hop_dict' after the flip
        
        loop_dict: dict
            Modified 'loop_dict' after the flip

        """
        
        hop_dict = self.__hop_dict
        loop_dict = self.__loop_dict
        
        h_index_2_site = {hop_dict[key]['index']:key for key in hop_dict.keys()}
        for flip in flips:
            hop_dict[h_index_2_site[flip]]['strength'] = - hop_dict[h_index_2_site[flip]]['strength']
            loop_indices = hop_dict[h_index_2_site[flip]]['loop_indices']
            for loop in loop_indices:
                loop_dict[loop]['w_L'] = -loop_dict[loop]['w_L']
                
        return hop_dict, loop_dict
    
    
    
    def vortex_pair(self, l1, l2, N_loops):
        """
        Creates a pair of vortices in loops l1 and l2 by flipping 'hop_strength' on the appropriate
        series of bonds.

        Parameters
        ----------
        l1 : int
            Index of the source loop.
        l2 : int
            Index of the destination loop.
        N_loops : int
            Number of loops in the structure.

        Returns
        -------
        hop_dict : dict
            modified hop_dict
        loop_dict : dict
            modified loop_dict

        """
        hop_dict = self.__hop_dict
        loop_dict = self.__loop_dict
        hop_index_dict = {hop_dict[key]['index']:key for key in hop_dict.keys()}
        graph_loops = self.__loop_dict_2_graph__(N_loops)
        path = self.__shortest_path_between_pair__(graph_loops, l1, l2)
        flips = self.__loop_path_to_flip__(path)
        N_flips = len(flips)
        
        #### Update w_L values    
        for index in [l1,l2]:
            loop_dict[index]['w_L'] = -loop_dict[index]['w_L']
        
        
        #### Update params and hamiltonian  
        for flip in flips:
            hop_dict[hop_index_dict[flip]]['strength'] = - hop_dict[hop_index_dict[flip]]['strength']
            
        return hop_dict, loop_dict, N_flips

    
    def number_of_flips_per_vortex_pair(self, l1, l2, N_loops):
        """Calculates the number of flips required to create an anyon pair on loop l1 to loop l2.

        Parameters
        ----------
        l1 : int
            Index of the source loop.
        l2 : int
            Index of the destination loop.
        N_loops : int
            Number of loops in the structure.

        Returns
        -------
        number_of_flips: list_
            Number of flips required to transport an anyon from loop l1 to loop l2.
        """
        hop_dict = self.__hop_dict
        loop_dict = self.__loop_dict
        hop_index_dict = {hop_dict[key]['index']:key for key in hop_dict.keys()}
        graph_loops = self.__loop_dict_2_graph__(N_loops)
        path = self.__shortest_path_between_pair__(graph_loops, l1, l2)
        flips = self.__loop_path_to_flip__(path)
        number_of_flips = len(flips)
        return number_of_flips

    
    
    
    
    def __loop_dict_2_graph__(self, N_loops):
        """
        Given the loop_dict of a system, this method returns a graph (list of lists) where graph[i] 
        contains the list of indices of the loops adjacent to the loop with index i.
        
        adjacent = share an edge.

        Parameters
        ----------
        N_loops : int
            Number of loops in the loop_dict 
            (in case loop_dict contains some additional info apart from the loops info)

        Returns
        -------
        loop_graph : list of lists
            A list of lists which should be treated as the adjacency matrix for the loops in the structure.

        """
        hop_dict = self.__hop_dict
        loop_dict = self.__loop_dict
        loop_graph = [[] for i in range(N_loops)]
        for i in range(N_loops):
            edges = loop_dict[i]['edges']
            for edge in edges:
                edge = edge['hop_sites']
                neighbor = deepcopy(hop_dict[edge]['loop_indices'])
                neighbor.remove(i)
                
                
                if neighbor!=[]:
                    assert len(neighbor)<=1, f'More than two loops are shared by single edge. len:{len(neighbor)}'
                    loop_graph[i].append(neighbor[0])
        return loop_graph
    
    
    def __BFS__(self, adj, src, dest, pred, dist):
        """
        Perform a breadth-first search to find the shortest path between the source vertex and destination
        vertex  in a unweighted and undirected graph.
        (Of course the source and the destination can be exchanged here as the graph is undirected)

        Parameters
        ----------
        adj : list of lists
            Adjacency matrix of the graph.
        src : int
            Index of the source vertex.
        dest : int
            Index of the destination vertex
        pred : list
            A list which contains the predecessor of the vertex visited.
        dist : list
            List containing the distance of the every vertex from the source.
            (These distances are not the shortest distance except for that of the destination)
        
        NOTES:1.This method is to be used in conjuction with the method: __shortest_path_between_pair__
              2.pred' and 'dist' get directly updated. 
        
        Returns
        -------
        bool
            True: if there exists a path to reach the destination.
            False: if path does not exists between the source and the destination.
        """
        queue = []
        v = len(adj)
        visited = [False for i in range(v)]
        for i in range(v):
            dist[i] = inf
            pred[i] = -1
            
        visited[src] = True
        dist[src] = 0
        queue.append(src)
        
        while (len(queue)!=0):
            u = queue[0]
            queue.pop(0)
            for i in range(len(adj[u])):
                if (visited[adj[u][i]] == False):
                    visited[adj[u][i]] = True
                    dist[adj[u][i]] = dist[u] + 1
                    queue.append(adj[u][i])
                    pred[adj[u][i]] = u
                   
                    if (adj[u][i]== dest):
                        return True
        return False
    
    
    
    def __shortest_path_between_pair__(self, adj, src, dest):
        """
        Finds shortest path between pair in a unweighted undirected graph  using the method __BFS__.

        Parameters
        ----------
        adj : list of lists
            Adjacency matrix of the list
        src : int
            Index of the source vertex.
        dest : int
            Index of the destination vertex.

        Returns
        -------
        path: list
            The list of vertices needs to be visited to reach the destination in shortest distance from the 
            source.

        """
        v = len(adj)
        pred = [0 for i in range(v)]
        dist = [0 for i in range(v)]
        
        if self.__BFS__(adj, src, dest, pred, dist) == False:
            print(f'Source vertex:{src} and destination vertex:{dest} are not connected.')
            raise Exception("Sorry, source and destination must be connected") 
            
        else:
            path = []
            crawl = dest
            path.append(crawl)
            
            while (pred[crawl] != -1):
                path.append(pred[crawl])
                crawl = pred[crawl]
            
            return path[::-1]
    
    
    def __loop_path_to_flip__(self, path):
        """
        Given a path (list of loop indices), this returns a list of the indices of the edge such that 
        return_list[i] = index of edge shared between path[i] and path[i+1]
        

        Parameters
        ----------
        path : list
            List of loop indices such that path[i] and path[i+1] are adjacent (share an edge)

        Returns
        -------
        flips : list
            List of shared edges.

        """
        hop_dict = self.__hop_dict
        loop_2_edge = {tuple(hop_dict[key]['loop_indices']):hop_dict[key]['index'] for key in hop_dict.keys()}
        N_edge = len(path)-1
        flips = []
        for i in range(N_edge):
            try:
                loop_indices =  (path[i], path[i+1])
                flip_index = loop_2_edge[loop_indices]
            except:
                loop_indices =  (path[i+1], path[i])
                flip_index = loop_2_edge[loop_indices]
            flips.append(flip_index)
            
        return flips
    
    
    def __flips_to_loops__(self, flips, N_loops):
        hop_dict = self.__hop_dict
        loop_dict = deepcopy(self.__loop_dict)
        hop_index_dict = {hop_dict[key]['index']:key for key in hop_dict.keys()}
        for flip in flips:
            loop_indices = hop_dict[hop_index_dict[flip]]['loop_indices']
            for index in loop_indices:
                loop_dict[index]['w_L'] = -loop_dict[index]['w_L']
        loops=[]
        for i in range(N_loops):
            if loop_dict[i]['w_L'] == -1:
                loops.append(i)
        return loops
        
        
        
    
    
    

# %% Kitaev honeycomb model
class Kitaev_honeycomb(__model_system__):
    
    def __init__(self, syst_params, hop_params):
        self.syst_params = syst_params
        self.hop_params = hop_params
        self.__hop_types = list(hop_params.keys())
        self.__hop_dict, self.__sites, self.N_sites, self.N_bonds, self.Nz = self.__hopping_dict__()
        self.__loop_dict, self.N_loops = self.__index_loops__()
        __model_system__.__init__(self, syst_params, hop_params, self.__sites, self.__hop_dict, self.__loop_dict)
        
    
    
    
    def __hopping_dict__(self):
        """
        Creates the hopping list of the translationally invariant system for the 
        Kitaev Honeycomb model.

        
        Returns
        -------
        hop_dict: dict of dict
            dictionary of hopping elements

        """
        L = self.syst_params['L']
        hop_params = self.hop_params
        pbc = self.syst_params['pbc']
        prim_vecs = [(1,0), (sin(pi/6), cos(pi/6))]
        a = 1/sqrt(27)
        basis = [(0,0), (0,3*a)]
        lat = kwant.lattice.Polyatomic(prim_vecs, basis, norbs=1)
        lat_a, lat_b =lat.sublattices
        
        
        def region(pos):
            """
            Creates a region (a square region of length L for now) in the system. Takes akwant.site as
            input and checks if it is in the square region of length L.

            Parameters
            ----------
            site : pos
                Position of the kwant site

            Returns
            -------
            1 if the site in the square, else 0.

            """
            x,y = pos
            return (x>-L and x<L) and (y>-L and y<L)
        
        syst = kwant.Builder()
        
        L = 2*int(L/2)
        for j in range(L):
            for i in range(-int(j/2),L-int(j/2)):
                tag = (i,j)
                for family in lat.sublattices:
                    site = kwant.builder.Site(family, tag)
                    syst[site] = 0
        
        hop_z = ((0,0),lat_a,lat_b)
        hop_x = ((0,1),lat_a,lat_b)
        hop_y = ((-1, 1),lat_a,lat_b)
        syst[kwant.builder.HoppingKind(*hop_z)] = hop_params['Jz']
        syst[kwant.builder.HoppingKind(*hop_x)] = hop_params['Jx']
        syst[kwant.builder.HoppingKind(*hop_y)] = hop_params['Jy']
        
        if pbc==True:
            #### PBC in y-dir
            j_N = L-1
            for i in range(-int(j_N/2),L-int(j_N/2)):
                tag2 = (i,j_N)
                site2 = kwant.builder.Site(lat_b, tag2)
                
                k = i+int(j_N/2)
                tag1y = (k,0)
                site1y = kwant.builder.Site(lat_a, tag1y)
                syst[site1y,site2] = hop_params['Jy']
                
                l = (k+1)%L
                if l==0:
                    break
                tag1x = (l,0)
                site1x = kwant.builder.Site(lat_a, tag1x)
                syst[site1x,site2] = hop_params['Jx']
                
            #### PBC in x-dir
            for j in range(0,L,2):
                if (j+1)%L == 0:
                    break
                i0 = -int(j/2)
                iN = L - int(j/2) - 1
                tag2y = (i0, j)
                site2y = kwant.builder.Site(lat_b, tag2y)
                
                
                tag1 = (iN, j+1)
                site1y = kwant.builder.Site(lat_a, tag1)
                syst[site1y,site2y] = hop_params['Jy']
                
                if (j+2)%L == 0:
                    break
                tag2x = (i0-1, j+2)
                site2x = kwant.builder.Site(lat_a, tag2x)
                site1x = kwant.builder.Site(lat_b, tag1)
                syst[site2x,site1x] = hop_params['Jx']
                
            
            site1 = kwant.builder.Site(lat_a, (0,0))
            site2 = kwant.builder.Site(lat_b, (L - int(j_N/2) - 1, L-1))
            syst[site1,site2] = hop_params['Jx']
            
        else:
            syst.eradicate_dangling()
        
        
        
        Nz = 0
        hop_dict={}
        for (i,hop) in enumerate(list(syst.hoppings())):
            site1 = hop[0]
            site2 = hop[1]
            fam1 = site1.family
            fam2 = site2.family
            tag1 = site1.tag
            tag2 = site2.tag
            if (fam1==lat_a and fam2==lat_b) and (tag1 == tag2):
                hop_type = 'Jz' # for Jz links
                hop_strength = -1 * hop_params['Jz']
                u_jk = 1
                u_index = Nz
                Nz = Nz + 1
            elif (fam1==lat_a and fam2==lat_b) and (tag1 - tag2 == [0,1]):
                hop_type = 'Jx' # for Jx bonds
                hop_strength = -1 * hop_params['Jx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_b) and (tag1- tag2 == [-1,1]):
                hop_type = 'Jy' #for Jy bonds
                hop_strength = hop_params['Jy']
                u_jk = nan
                u_index = nan
            
            #### PBC conditions
            elif (fam1==lat_a and fam2==lat_b) and (tag1- tag2 == [int(j_N/2)+1,-j_N]):
                hop_type = 'Jx'
                hop_strength = -1 * hop_params['Jx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_b) and (tag1- tag2 == [int(j_N/2),-j_N]):
                hop_type = 'Jy'
                hop_strength = hop_params['Jy']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_b) and (tag2- tag1 == [L,-1]):
                hop_type = 'Jx'
                hop_strength = -1 * hop_params['Jx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_b) and (tag1- tag2 == [L-1,1]):
                hop_type = 'Jy'
                hop_strength = hop_params['Jy']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_b) and (tag1- tag2 == [-(L - int(j_N/2) - 1),-L+1]):
                hop_type = 'Jx'
                hop_strength = -1 * hop_params['Jx']
                u_jk = nan
                u_index = nan
                
            else:
                assert 0 > 1, f'hop_type:{fam1, fam2} does not belong to any of the classes' 
                
                    
            hop_dict[(site1,site2)] = {'type':hop_type, 'strength':hop_strength, 'index':i, 'loop_indices':[], 'u_jk':u_jk, 'u_index':u_index}
        
        sites = syst.finalized().sites
        N_sites = syst.finalized().graph.num_nodes
        N_bonds = len(hop_dict)
        Nz = Nz
        
        return hop_dict, sites, N_sites, N_bonds, Nz



    def __index_loops__(self):
        """
        Create an indexing for the fundamental cycles present in Honeycomb lattice. Store it in the
        form of a dictionary which has the index of the loop as the key and list of edges of the loop
        as the value.

        Parameters
        ----------
        hop_dict : dict
            Hopping dictionary created for Sg-3 by the function 'hopping_list_SG3'

        Returns
        -------
        loop_dict: dict
            Dictionary containing the {index:[ list of edges]} as the key value pair

        """
        hop_dict = self.__hop_dict    
        
        hop_keys = list(hop_dict.keys())
        
        fam_a = hop_keys[0][0].family
        fam_b = hop_keys[0][1].family
        
        hop_index_dict = {hop_dict[key]['index']:{'hop_sites':key, 'hop_strength':hop_dict[key]['strength'], 'hop_type':hop_dict[key]['type'], 'hop_index':hop_dict[key]['index']} for key in hop_keys}
        loop_dict = {}
        
        def loop(i,j):
            """
            Create an element of the loop_dict which is dedicated for the unit cell defined by (i,j).

            Parameters
            ----------
            i, j : int
                Integers defining the unit cell. The position vector for the unit cell is given by 
                i*v1 + j*v2 if v1 and v2 are the translation vectors.

            Returns
            -------
            loop_el: list
            """
            families = [fam_a, fam_b, fam_a, fam_b, fam_a, fam_b]
            tags = [[0,0], [0,0], [0,1], [1,0], [1,0], [1,-1]]
            tags = [[tag[0]+i, tag[1]+j] for tag in tags]
            loop_el = []
            for k in range(6):
                l = (k+1)%6
                site1 = kwant.builder.Site(families[k], tags[k])
                site2 = kwant.builder.Site(families[l], tags[l])
                try:
                    index = hop_dict[(site1,site2)]['index']
                except KeyError:
                    try:
                        index = hop_dict[(site2,site1)]['index']
                    except KeyError:
                        loop_el = []
                        break
                loop_el.append(hop_index_dict[index])
            return loop_el
        
        tags = []
        for key in hop_keys:
            site1 = key[0]
            site2 = key[1]
            tags.append(site1.tag)
            tags.append(site2.tag)
        tags = list(set(tags))
        # print(f'N_tags: {len(tags)}')
        
        N_loop = 0
        for tag in tags:
            i = tag[0]
            j = tag[1]
            loop_el = loop(i,j)
            if len(loop_el)!=0:
                loop_dict[N_loop] = loop_el
                for edge in loop_el:
                    hop_dict[edge['hop_sites']]['loop_indices'].append(N_loop)
                N_loop = N_loop + 1
        
        loop_dict = {k:{'edges':loop_dict[k], 'w_L':1, 'index':k} for k in loop_dict.keys()}
        
        for key in hop_dict.keys():
            hop_dict[key]['loop_indices'] = list(set(hop_dict[key]['loop_indices']))
        
        return loop_dict, N_loop
        



# %% Yao Kivelson Model
        
class Yao_Kivelson(__model_system__):
    
    def __init__(self, syst_params, hop_params):         
        self.syst_params = syst_params
        self.hop_params = hop_params
        self.__hop_types = list(hop_params.keys())
        self.__hop_dict, self.__sites, self.N_sites, self.N_bonds, self.Nz = self.__hopping_dict__()
        self.__loop_dict, self.N_loops = self.__index_loops__()
        __model_system__.__init__(self, syst_params, hop_params, self.__sites, self.__hop_dict, self.__loop_dict)
    
    
        
    def __hopping_dict__(self):
        """
        Creates the hopping list of the translationally invariant system (YAO_Kivelson_model).

        Parameters
        ----------
        L: int
            Length of the square region
        hop_params: dict
            Parameter dictionary for the model

        Returns
        -------
        hop_dict: dict of dict
            dictionary of hopping elements

        """
        L = self.syst_params['L']
        hop_params = self.hop_params
        pbc = self.syst_params['pbc']
        
        prim_vecs = [(1,0), (sin(pi/6), cos(pi/6))]
        a = 1/(2+sqrt(3))
        basis = [(0,a/sqrt(3)), (0,a+a/sqrt(3)), (a/sqrt(3)*cos(pi/6), a+2*a/sqrt(3)+a/sqrt(3)*sin(pi/6)), ((a+a/sqrt(3))*cos(pi/6), a+2*a/sqrt(3)+(a+a/sqrt(3))*sin(pi/6)), (-a/sqrt(3)*cos(pi/6), a+2*a/sqrt(3)+a/sqrt(3)*sin(pi/6)), (-(a+a/sqrt(3))*cos(pi/6), a+2*a/sqrt(3)+(a+a/sqrt(3))*sin(pi/6)) ]
        lat = kwant.lattice.Polyatomic(prim_vecs, basis, norbs=1)
        lat_a, lat_b, lat_c, lat_d, lat_e, lat_f =lat.sublattices
        
        def region(pos):
            """
            Creates a region (a square region of length L for now) in the system. Takes akwant.site as
            input and checks if it is in the square region of length L.

            Parameters
            ----------
            site : pos
                Position of the kwant site

            Returns
            -------
            1 if the site in the square, else 0.

            """
            x,y = pos
            return (x>-L and x<L) and (y>-L and y<L)
        
        syst = kwant.Builder()
        
        L = 2*int(L/2)
        for j in range(L):
            for i in range(-int(j/2),L-int(j/2)):
                tag = (i,j)
                for family in lat.sublattices:
                    site = kwant.builder.Site(family, tag)
                    syst[site] = 0
        
        hop_Jz = ((0,0),lat_a,lat_b)
        hop_Jy = ((0,0),lat_c,lat_d)
        hop_Jx = ((0,0),lat_f,lat_e)
        hop_tx = [((0,0),lat_b,lat_c), ((0,1),lat_a,lat_d)]
        hop_ty = [((0,0),lat_e,lat_b), ((-1,1),lat_a,lat_f)]
        hop_tz = [((0,0),lat_c,lat_e), ((1,0),lat_f,lat_d)]
        syst[[kwant.builder.HoppingKind(*hop) for hop in hop_tz]] = hop_params['tz']
        syst[[kwant.builder.HoppingKind(*hop) for hop in hop_tx]] = hop_params['tx']
        syst[[kwant.builder.HoppingKind(*hop) for hop in hop_ty]] = hop_params['ty']
        syst[kwant.builder.HoppingKind(*hop_Jz)] = hop_params['Jz']
        syst[kwant.builder.HoppingKind(*hop_Jx)] = hop_params['Jx']
        syst[kwant.builder.HoppingKind(*hop_Jy)] = hop_params['Jy']
        
        
        if pbc==True:
            #### PBC in y-dir
            j_N = L-1
            for i in range(-int(j_N/2),L-int(j_N/2)):
                tag2 = (i,j_N)
                site2y = kwant.builder.Site(lat_f, tag2)
                site2x = kwant.builder.Site(lat_d, tag2)
                
                
                k = i+int(j_N/2)
                tag1y = (k,0)
                site1y = kwant.builder.Site(lat_a, tag1y)
                syst[site1y,site2y] = hop_params['ty']
                
                l = (k+1)%L
                if l==0:
                    break
                tag1x = (l,0)
                site1x = kwant.builder.Site(lat_a, tag1x)
                syst[site1x,site2x] = hop_params['tx']
                # print(f'k:{k}, l:{l}')
                
            #### PBC in x-dir
            for j in range(0,L,2):
                i0 = -int(j/2)
                iN = L - int(j/2) - 1
                
                site1z = kwant.builder.Site(lat_f, (i0,j))
                site2z = kwant.builder.Site(lat_d, (iN,j))
                syst[site1z, site2z] = hop_params['tz']
                
                if (j+1)%L == 0:
                    break
                tag2y = (i0, j)
                site2y = kwant.builder.Site(lat_f, tag2y)
                
                
                tag1 = (iN, j+1)
                site1y = kwant.builder.Site(lat_a, tag1)
                syst[site1y,site2y] = hop_params['ty']
                
                
                if (j+2)%L == 0:
                    break
                tag2x = (i0-1, j+2)
                site2x = kwant.builder.Site(lat_a, tag2x)
                site1x = kwant.builder.Site(lat_d, tag1)
                syst[site2x,site1x] = hop_params['tx']
                
                site1z = kwant.builder.Site(lat_f, (i0,j+1))
                site2z = kwant.builder.Site(lat_d, (iN,j+1))
                syst[site1z, site2z] = hop_params['tz']
                
                
                # diff = [tag1[0] - tag2x[0], tag1[1] - tag2x[1] ]
                # print(f'tag2x:{tag2x}, tag1:{tag1}, i0:{i0}, iN:{iN}, diff:{diff}, L:{L}')
            
                
            site1z = kwant.builder.Site(lat_f, (-int((L-1)/2),L-1))
            site2z = kwant.builder.Site(lat_d, (L - int((L-1)/2) - 1,L-1))
            syst[site1z, site2z] = hop_params['tz']
                
            site1 = kwant.builder.Site(lat_a, (0,0))
            site2 = kwant.builder.Site(lat_d, (L - int(j_N/2) - 1, L-1))
            syst[site1,site2] = hop_params['Jx']
        
        else:
            syst.eradicate_dangling()
        
        Nz = 0
        hop_dict={}
        for (i,hop) in enumerate(list(syst.hoppings())):
            site1 = hop[0]
            site2 = hop[1]
            fam1 = site1.family
            fam2 = site2.family
            tag1 = site1.tag
            tag2 = site2.tag
            if fam1==lat_a and fam2==lat_b:
                hop_type = 'Jz' # for Jz links
                hop_strength = -1 * hop_params['Jz']
                u_jk = 1
                u_index = Nz
                Nz = Nz + 1
            elif fam1==lat_f and fam2==lat_e:
                hop_type = 'Jx' # for Jx bonds
                hop_strength = -1 * hop_params['Jx']
                u_jk = nan
                u_index = nan
            elif fam1==lat_c and fam2==lat_d:
                hop_type = 'Jy' #for Jy bonds
                hop_strength = hop_params['Jy']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_c and fam2==lat_e) or (fam1==lat_f and fam2==lat_d):
                hop_type = 'tz' # for tz links
                hop_strength = -1 * hop_params['tz']
                u_jk = 1
                u_index = Nz
                Nz = Nz + 1
            elif (fam1==lat_b and fam2==lat_c) or (fam1==lat_a and fam2==lat_d):
                hop_type = 'tx' # for tx bonds
                hop_strength = -1 * hop_params['tx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_e and fam2==lat_b) or (fam1==lat_a and fam2==lat_f):
                hop_type = 'ty' #for ty bonds
                hop_strength = hop_params['ty']
                u_jk = nan
                u_index = nan
                
            #### PBC conditions
            elif (fam1==lat_a and fam2==lat_d) and (tag1- tag2 == [int(j_N/2)+1,-j_N]):
                hop_type = 'tx'
                hop_strength = -1 * hop_params['tx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_f) and (tag1- tag2 == [int(j_N/2),-j_N]):
                hop_type = 'ty'
                hop_strength = hop_params['ty']
                u_jk = nan
                u_index = nan
                
            elif (fam1==lat_a and fam2==lat_f) and (tag2- tag1 == [L,-1]):
                hop_type = 'tx'
                hop_strength = -1 * hop_params['tx']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_a and fam2==lat_f) and (tag1- tag2 == [L-1,1]):
                hop_type = 'ty'
                hop_strength = hop_params['ty']
                u_jk = nan
                u_index = nan
            elif (fam1==lat_f and fam2==lat_d) and (tag1- tag2 == [-(L-1),0]):
                hop_type = 'tz'
                hop_strength = -1 * hop_params['tz']
                u_jk = 1
                u_index = Nz
                
            elif (fam1==lat_a and fam2==lat_d) and (tag1- tag2 == [-(L - int(j_N/2) - 1),-L+1]):
                hop_type = 'tx'
                hop_strength = -1 * hop_params['tx']
                u_jk = nan
                u_index = nan
            else:
                assert 0 > 1, f'hop_type:{fam1, fam2} does not belong to any of the classes' 
            hop_dict[(site1,site2)] = {'type':hop_type, 'strength':hop_strength, 'index': i, 'loop_indices':[], 'u_jk':u_jk, 'u_index':u_index}
        
        sites = syst.finalized().sites
        N_sites = syst.finalized().graph.num_nodes
        N_bonds = len(hop_dict)
        Nz = Nz
        
        return hop_dict, sites, N_sites, N_bonds, Nz
    
    
    
    def __index_loops__(self):
        """
        Create an indexing for the fundamental cycles present in Decorated Honeycomb lattice.
        Store it in the form of a dictionary which has the index of the loop as the key 
        and list of edges of the loop as the value.

        Parameters
        ----------
        hop_dict : dict
            Hopping dictionary created for Sg-3 by the function 'hopping_list_SG3'

        Returns
        -------
        loop_dict: dict
            Dictionary containing the {index:[ list of edges]} as the key value pair

        """
        
        hop_dict = self.__hop_dict
        
        prim_vecs = [(1,0), (sin(pi/6),cos(pi/6))]
        a = 1/(2+sqrt(3))
        basis = [(0,a/sqrt(3)), (0,a+a/sqrt(3)), (a/sqrt(3)*cos(pi/6), a+2*a/sqrt(3)+a/sqrt(3)*sin(pi/6)), ((a+a/sqrt(3))*cos(pi/6), a+2*a/sqrt(3)+(a+a/sqrt(3))*sin(pi/6)), (-a/sqrt(3)*cos(pi/6), a+2*a/sqrt(3)+a/sqrt(3)*sin(pi/6)), (-(a+a/sqrt(3))*cos(pi/6), a+2*a/sqrt(3)+(a+a/sqrt(3))*sin(pi/6)) ]
        lat = kwant.lattice.Polyatomic(prim_vecs, basis, norbs=1)
        lat_a, lat_b, lat_c, lat_d, lat_e, lat_f =lat.sublattices
        
        
        hop_keys = list(hop_dict.keys())
        hop_index_dict = {hop_dict[key]['index']:{'hop_sites':key, 'hop_strength':hop_dict[key]['strength'], 'hop_type':hop_dict[key]['type'], 'hop_index':hop_dict[key]['index']} for key in hop_keys}
        loop_dict = {}
        
        def loop_yk(i,j,families, tags):
            """
            Create an element of the loop_dict which is dedicated for the unit cell defined by (i,j).

            Parameters
            ----------
            i, j : int
                Integers defining the unit cell. The position vector for the unit cell is given by 
                i*v1 + j*v2 if v1 and v2 are the translation vectors.

            Returns
            -------
            loop_el: list
            """
            assert len(families)==len(tags), f'Mismatch in number of site.families:{len(families)} and site.tags:{len(tags)}'
            
            tags = [[tag[0]+i, tag[1]+j] for tag in tags]
            loop_el = []
            N_fam = len(families)
            for k in range(N_fam):
                l = (k+1)%N_fam
                site1 = kwant.builder.Site(families[k], tags[k])
                site2 = kwant.builder.Site(families[l], tags[l])
                try:
                    index = hop_dict[(site1,site2)]['index']
                except KeyError:
                    try:
                        index = hop_dict[(site2,site1)]['index']
                    except KeyError:
                        loop_el = []
                        break
                loop_el.append(hop_index_dict[index])
            return loop_el
        
        tags = []
        for key in hop_keys:
            site1 = key[0]
            site2 = key[1]
            tags.append(site1.tag)
            tags.append(site2.tag)
        tags = list(set(tags))
        # print(f'N_tags: {len(tags)}')
        
        families_dodec = [lat_a, lat_b, lat_c, lat_d, lat_f, lat_e, lat_b, lat_a, lat_d, lat_c, lat_e, lat_f]
        tags_dodec = [[0,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0],[1,-1],[1,-1],[1,-1],[1,-1]]
        families_trig1 = [lat_b, lat_c, lat_e]
        tags_trig1 = [[0,0],[0,0],[0,0]]
        families_trig2 = [lat_a, lat_f, lat_d]
        tags_trig2 = [[0,0], [1,-1], [0,-1]]
        
        families_init = [families_dodec, families_trig1, families_trig2]
        tags_init = [tags_dodec, tags_trig1, tags_trig2]
        
        N_loop = 0
        for tag in tags:
            i = tag[0]
            j = tag[1]
            for n_l in range(3):
                loop_el = loop_yk(i,j, families_init[n_l], tags_init[n_l])
                if len(loop_el)!=0:
                    loop_dict[N_loop] = loop_el
                    for edge in loop_el:
                        hop_dict[edge['hop_sites']]['loop_indices'].append(N_loop)
                    N_loop = N_loop + 1
        
        loop_dict = {k:{'edges':loop_dict[k], 'w_L':1, 'index':k} for k in loop_dict.keys()}
        
        for key in hop_dict.keys():
            hop_dict[key]['loop_indices'] = list(set(hop_dict[key]['loop_indices']))
        
        return loop_dict, N_loop
      


# %% Kitaev SG3 model
class Kitaev_SG3(__model_system__):
    
    def __init__(self, syst_params, hop_params):
        self.syst_params = syst_params
        self.hop_params = hop_params
        self.__hop_types = list(hop_params.keys())
        self.__lat = kwant.lattice.triangular(1,norbs=1)
        self.__add_b_site = syst_params['add_b_site']
        
        self.__sites, self.N_sites = self.__site_list__()
        self.__hop_dict, self.N_bonds, self.Nz = self.__hopping_dict__()
        self.__loop_dict, self.N_loops = self.__index_loops__() 
        __model_system__.__init__(self, syst_params, hop_params, self.__sites, self.__hop_dict, self.__loop_dict)
        
        
    def __iter_site_SG3__(self, sites, g_prev):
        """
        Iteratively generates the sites for a self-similar SG3 of generation g from generation g-1.

        Parameters
        ----------
        sites : list of kwant.sites
            list of sites of SG-3 in the previous generation
        g_prev : int
            previous generation, g-1

        Returns
        -------
        sites_g : list of kwant.sites
            list of site sof SG-3 of the current generation, g

        """
        
        sites_g = []
        
        def element(site, i, j):
            ### Translates a site of a prev gen to the new position of the subsequent gen.
            g = g_prev + 1
            x = site.tag[0] + i*2**(g)
            y = site.tag[1] + j*2**(g)
            return self.__lat(x,y)
        
        sector_list = [[0,0],[0,1],[1,0]]
        str_list = [0,1,2]
          
        for site in sites:
            for index in str_list:
                i = sector_list[index][0]
                j = sector_list[index][1]
                new_site = element(site, i, j)
                sites_g.append(new_site)
                           
        return sites_g

    def __site_list__(self):
        """
        Function to generate the final usable list of sites of SG-3

        Parameters
        ----------
        G : int
            Generation of the SG-3 to be constructed.

        Returns
        -------
        sites : list of kwant.sites
            list of site sof SG-3 of the current generation, G        

        """
        G = self.syst_params['G']-1
        add_b_site = self.__add_b_site
        
        sites = [self.__lat(0,0), self.__lat(1,0), self.__lat(0,1)] # Create initial lattice points for recursion
        for g in range(G):
            sites = self.__iter_site_SG3__(sites, g)
        
        if add_b_site==True:
            tag_x_list = [ site.tag[0] for site in sites]
            tag_x_new = max(tag_x_list)+1
            new_site = self.__lat(tag_x_new, 0)
            sites.append(new_site)
        
        N_sites = len(sites)
        return sites, N_sites
    
    def __iter_hop_SG3__(self, hops, g_prev, u_index):
        """
        ### Iteratively generates the hopping dictionary for the self_similar SG3.

        Parameters
        ----------
        hops : list of dict
            list of dictionary of hopping elements. Each element of the list is of
            the following form- {'hop_sites':tuple of kwant.sites, 'hop_type': type of hopping, 'hop_strength': strength of hopping}
        g_prev : int
            previous generation, g-1.
        hop_param : dict
            Dictionary of parameters for the hopping elements of the full system.
            It is of the form: {'tx','ty', 'tz', 'Jx', 'Jy', 'Jz'.}

        Returns
        -------
        hops_g: list of dict
            list of dictionary of hopping elements of the current generation. Format is same as hops.

        """
        
        g = g_prev + 1
        hops_g = []
        hop_types = self.__hop_types
        hop_indices = {key: index for (index, key) in enumerate(self.__hop_types)}
        u_index = 0
               
        def element(site, i, j):
            ### Translates a site of a prev gen to the new position of the subsequent gen.
            ### Input: site: kwant_site, i: int, j: int
            ### Output: kwant_site
            g = g_prev + 1
            x = site.tag[0] + i*2**(g)
            y = site.tag[1] + j*2**(g)
            return self.__lat(x,y)
        
        sector_list = [[0,0],[0,1],[1,0]]
        str_list = [0,1,2]
        
                      
        for index in str_list:
            i = sector_list[index][0]
            j = sector_list[index][1]
            
            
            for hop in hops:
                site1 = hop['hop_sites'][0]
                site2 = hop['hop_sites'][1]
                hop_type = hop['hop_type']
                assert hop_type == 'tx' or hop_type == 'ty' or hop_type == 'tz' or hop_type == 'Jx' or hop_type == 'Jy' or hop_type == 'Jz', f'Wrong hop_type:{hop_type}'
                hop_index = hop_indices[hop_type]
                
                hop_dict = {}
                new_site1 = element(site1, i, j)
                new_site2 = element(site2, i, j)
            
                if hop_index > 2:
                    new_hop_index = (((hop_index-3) + index*2**g) % 3) + 3
                else:
                    new_hop_index = (hop_index + index*2**g) % 3 
                
                hop_dict['hop_type'] = hop_types[new_hop_index]
                if hop_types[new_hop_index]=='ty' or hop_types[new_hop_index]=='Jy':
                    hop_dict['hop_strength'] = self.hop_params[hop_types[new_hop_index]]
                    hop_dict['u_jk'] = nan
                    hop_dict['u_index'] = nan
                elif hop_types[new_hop_index]=='tx' or hop_types[new_hop_index]=='Jx':
                    hop_dict['hop_strength'] = (-1) * self.hop_params[hop_types[new_hop_index]]
                    hop_dict['u_jk'] = nan
                    hop_dict['u_index'] = nan
                elif hop_types[new_hop_index]=='tz' or hop_types[new_hop_index]=='Jz':
                    hop_dict['hop_strength'] = (-1) * self.hop_params[hop_types[new_hop_index]]
                    hop_dict['u_jk'] = 1
                    hop_dict['u_index'] = u_index
                    u_index = u_index + 1
                else:
                    assert 0>1, f"Incorrect hop_type:{hop_dict['hop_type']} detected"
                
                hop_dict['hop_sites'] = (new_site1, new_site2)
                hops_g.append(hop_dict)
        
        ## Extra hoppings
        T = array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        str_list=[1,2,0]
        for index in str_list:
            hop_dict = {}
            i = sector_list[index][0]
            j = sector_list[index][1]
            new_site1 = self.__lat((1-i)*2**(g) - (1-i)*0, (1-j)*2**(g) - (1-j)*1)
            new_site2 = self.__lat((1-i)*2**(g) - (1-i)*1, (1-j)*2**(g) - (1-j)*0)
            v = zeros(3)
            v[index] = 1
            if g % 2 == 0:
                v = T @ v
            new_index = (nonzero(v)[0][0] + 1) % 3 + 3
            hop_dict['hop_type'] = hop_types[new_index]
            if hop_dict['hop_type']=='ty' or hop_dict['hop_type']=='Jy':
                hop_dict['hop_strength'] = self.hop_params[hop_dict['hop_type']]
                hop_dict['u_jk'] = nan
                hop_dict['u_index'] = nan
            elif hop_dict['hop_type']=='tx' or hop_dict['hop_type']=='Jx':
                hop_dict['hop_strength'] = (-1) * self.hop_params[hop_dict['hop_type']]
                hop_dict['u_jk'] = nan
                hop_dict['u_index'] = nan
            elif hop_dict['hop_type']=='tz' or hop_dict['hop_type']=='Jz':
                hop_dict['hop_strength'] = (-1) * self.hop_params[hop_dict['hop_type']]
                hop_dict['u_jk'] = 1
                hop_dict['u_index'] = u_index
                u_index = u_index + 1
            else:
                assert 0>1, f"Incorrect hop_type:{hop_dict['hop_type']} detected"
            hop_dict['hop_sites'] = (new_site1, new_site2)
            hops_g.append(hop_dict)
        
        
        return hops_g, u_index
    
    def __hopping_dict__(self):
        """
        Generates the full hopping list for SG-3 of generation G.

        Parameters
        ----------
        G : int
            Generation of SG-3.
        hop_param : dict
            Dictionary of hopping parametes.

        Returns
        -------
        hops : list of dict
            Complete list of hopping dictiponary elements of SG-3 of generation G.
        hop_dict : Dict
            Same data as hops but in alternative format. 
            Format: {'hop_sites':{'type':hop_types, 'strength': hop_strength]} for hop in hops}

        """
        
        hop_types = self.__hop_types
        hop_params = self.hop_params

        trig = self.__lat
        G = self.syst_params['G']-1
        
        hop_u_init = {}
        hop_u_init['tz'] = {'u_jk':1, 'u_index':0}
        hop_u_init['tx'] = {'u_jk':nan, 'u_index':nan}
        hop_u_init['ty'] = {'u_jk':nan, 'u_index':nan}
        
        hops = [{'hop_type': hop_types[0], 'u_jk': hop_u_init[hop_types[0]]['u_jk'], 'u_index': hop_u_init[hop_types[0]]['u_index'], 'hop_strength': hop_params[hop_types[0]], 'hop_sites': (trig(0,0), trig(1,0))}, 
                {'hop_type': hop_types[2], 'u_jk': hop_u_init[hop_types[2]]['u_jk'], 'u_index': hop_u_init[hop_types[2]]['u_index'], 'hop_strength': hop_params[hop_types[2]], 'hop_sites': (trig(0,1), trig(0,0))}, 
                {'hop_type': hop_types[1], 'u_jk': hop_u_init[hop_types[1]]['u_jk'], 'u_index': hop_u_init[hop_types[1]]['u_index'], 'hop_strength': hop_params[hop_types[1]], 'hop_sites': (trig(1,0), trig(0,1))}]
            
        u_index =1
        for g in range(G):
            hops, u_index = self.__iter_hop_SG3__(hops, g, u_index)
            
        hop_dict = {hop['hop_sites']:{'type':hop['hop_type'], 'strength':hop['hop_strength'], 'index':i, 'loop_indices':[], 'u_jk':hop['u_jk'], 'u_index':hop['u_index']} for (i,hop) in enumerate(hops)}
        
        N_bonds = len(hop_dict)
        Nz = round(N_bonds/3)
        assert Nz==u_index, f'Nz:{u_index} != Nz from N_bonds:{Nz}'
        return hop_dict, N_bonds, Nz
    
    
    
    def __iter_loops_SG3__(self, loops_prev, hop_index_dict):
        """
        Iteratively generates the loops dictionary for the self_similar SG3.

        Parameters
        ----------
        loops_prev : dict
            Dictionary of the loops for the previous generation of SG3.
        hop_index_dict : dict
            Dictionary which contains the followinf info {hop_dict['index']:{sites, strength, type, index}}

        Returns
        -------
        loops_g : dict
            Dictionary of the loops for the present generation.
        """
        
        N_L = len(loops_prev)-3
        G = round(log(2*N_L + 1)/log(3))
        N_bonds = round(3*(3**G-1)/2)
        loops_g = {}
        
        #### Recreate existing loops
        for j in range(3):
            for i in range(N_L):
                indices = [loops_prev[i][n]['hop_index'] for (n,ed) in enumerate(loops_prev[i])]
                hop_indices = [index + j*N_bonds  for index in indices]
                loops_g[i + j*N_L] = [hop_index_dict[hop_index] for hop_index in hop_indices]
                [self.__hop_dict[hop_index_dict[hop_index]['hop_sites']]['loop_indices'].append(i + j*N_L) for hop_index in hop_indices]
                
        #### Add the additional loop
        indices_l = [loops_prev['l'][n]['hop_index'] for (n,ed) in enumerate(loops_prev['l'])] 
        indices_r = [loops_prev['r'][n]['hop_index'] for (n,ed) in enumerate(loops_prev['r'])]
        indices_h = [loops_prev['h'][n]['hop_index'] for (n,ed) in enumerate(loops_prev['h'])]
        hop_l = [hop_index_dict[index + 2*N_bonds] for index in indices_l]
        hop_r = [hop_index_dict[index] for index in indices_r]
        hop_h = [hop_index_dict[index + N_bonds] for index in indices_h]
        
        loops_g[3*round((3**G-1)/2)] = [*hop_l, hop_index_dict[3*N_bonds], *hop_r, hop_index_dict[3*N_bonds+1], *hop_h,  hop_index_dict[3*N_bonds+2]]
        hop_indices = [3*N_bonds, 3*N_bonds+1, 3*N_bonds+2]
        [hop_indices.append(index) for index in indices_r]
        [hop_indices.append(index + N_bonds) for index in indices_h]
        [hop_indices.append(index + 2*N_bonds) for index in indices_l]
        [self.__hop_dict[hop_index_dict[hop_index]['hop_sites']]['loop_indices'].append(3*round((3**G-1)/2)) for hop_index in hop_indices]
        
        
        
        loops_g['h'] = [*[hop_index_dict[index] for index in indices_h], hop_index_dict[3*N_bonds], *[hop_index_dict[index + 2*N_bonds] for index in indices_h]]
        loops_g['l'] = [*[hop_index_dict[index] for index in indices_l], hop_index_dict[3*N_bonds+1], *[hop_index_dict[index + N_bonds] for index in indices_l]]
        loops_g['r'] = [*[hop_index_dict[index + N_bonds] for index in indices_r], hop_index_dict[3*N_bonds+2], *[hop_index_dict[index + 2*N_bonds] for index in indices_r]]        
        
        return loops_g
    
    
    
    def __index_loops__(self):
        """
        Create an indexing for the fundamental cycles present in SG-3. Store it in the form of a 
        dictionary which has the index of the loop as the key and list of edges of the loop as the value.

        Parameters
        ----------
        hop_dict : dict
            Hopping dictionary created for Sg-3 by the function 'hopping_list_SG3'

        Returns
        -------
        loop_dict: dict
            Dictionary containing the {index:[ list of edges]} as the key value pair

        """
        hop_dict = self.__hop_dict
        hop_keys = list(hop_dict.keys())
        G = self.syst_params['G']
        
        
        hop_index_dict = {hop_dict[key]['index']:{'hop_sites':key, 'hop_strength':hop_dict[key]['strength'], 'hop_type':hop_dict[key]['type'], 'hop_index':hop_dict[key]['index']} for key in hop_keys}
        loop_dict = {}
        
        loop_dict[0] = [hop_index_dict[0], hop_index_dict[1], hop_index_dict[2]]
        loop_dict['l'] = [hop_index_dict[1]]
        loop_dict['r'] = [hop_index_dict[2]]
        loop_dict['h'] = [hop_index_dict[0]]
        
        
        
        #### Create loop_dict
        for i in range(G-1):
            loop_dict = self.__iter_loops_SG3__(loop_dict, hop_index_dict)     
       
        
        loop_dict = {k:{'edges':loop_dict[k], 'w_L':1, 'index':k} for k in loop_dict.keys() }
        
        #### Update hop_dict
        for key in hop_dict.keys():
            hop_dict[key]['loop_indices'] = list(set(hop_dict[key]['loop_indices']))
        
        N_loops = len(loop_dict) - 3
        
        return loop_dict, N_loops