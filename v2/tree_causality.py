import glob
import os
from scipy.stats import pearsonr


import numpy as np
import pandas as pd

from tigramite import data_processing
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.plotting import plot_graph, plot_time_series_graph


class CausalPrecursors():
    """find causal precursors for Forest Causal LSTM (FCLSTM).

    The procedure is shown as following:
    1. get causality relationships based on PCMCI (Runge et al., 2020).
    2. plot PGM in time series types.
    3. group input variables that has the same causal activation time 
       on response variables.
    4. prune the causal link which pearson correlation isn't significant (p<0.05)
    5. for each group, generate tree causality-structure as CLSTM inputs.
       This input contains input variable index, number of child nodes, 
       state index for each node in causality tree.
    6. concentate above three index of all groups into three dictionaries.    

    Args:
        site_name (str, optional): [description]. Defaults to 'US-WCr'.
        cond_ind_test (str, optional): [description]. Defaults to 'parcorr'.
        max_tau (int, optional): [description]. Defaults to 7.
        sig_thres (float, optional): [description]. Defaults to 0.05.
        var_names (list, optional): [description]. Defaults to ['TA','P','TS','SM'].
        depth (int, optional): [description]. Defaults to 2.
        num_features (int, optional): [description]. Defaults to 4.

    Raises:
            ValueError: [description]
            ValueError: [description]
    """
    
    def __init__(
        self, 
        site_name='US-WCr',
        cond_ind_test='parcorr',
        max_tau=7,
        sig_thres=0.05,
        var_names=['TA','SW','LW','PA','P','WS','TS','SM'],
        depth=2, 
        num_features=8
        ):
        if cond_ind_test=='parcorr':
            self.cond_ind_test = ParCorr()
        elif cond_ind_test == '':
            raise ValueError('Not support yet!')

        if len(var_names) != num_features:
            raise ValueError('Give coincidence number and name of features!')
        
        self.site_name = site_name
        self.sig_thres = sig_thres
        self.max_tau = max_tau
        self.var_names = var_names
        self.depth = depth
        self.num_features = num_features

    def __call__(self, inputs):
        print('find group causal precursors for {} sites'.\
            format(self.site_name))
        
        print('1. get causaity based on PCMCI for {} features with {} window'.\
            format(self.num_features, self.max_tau))
        self.get_causal_precursors(inputs)

        print('2. plot causality relationships based on PGM')
        self.plot_causal_prescursors()

        print('3. group causal drivers for the same causal activation time')
        self.group_causal_prescursors()

        print('4. prune causal links')
        self.prune_causal_link(np.array(inputs))

        print('5. get group {} trees'.\
            format(len(self.causal_link_groups.keys())))
        self.get_group_trees()

        return self

    def get_causal_precursors(self, inputs):
        """get causal precursors used PCMCI."""
        dataframe = data_processing.DataFrame(np.array(inputs))
        
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test)
        
        self.pcmci.run_pcmci(
            tau_max=self.max_tau,
            pc_alpha=None)

    def plot_causal_prescursors(self):
        """plot causality relationships based on PGM."""
        plot_graph(
            val_matrix=self.pcmci.val_matrix, 
            sig_thres=self.sig_thres, 
            var_names=self.var_names,
            figsize=(10,10),
            save_name=self.site_name+'_causality_structure.pdf')
    
    def group_causal_prescursors(self):
        """group causal prescursors for the same causal time."""
        # handle val matrix get from pcmci,
        # and generate link matrix and impact matrix
        impact_matrix = np.abs(self.pcmci.val_matrix)
        link_matrix = impact_matrix >= self.sig_thres

        self.causal_link_groups = {}
        self.causal_impact_groups = {}

        for timestep in np.arange(self.max_tau):
            # link for each timestep of leaf node (set as SSM)
            link_t_leaf = link_matrix[:,-1,timestep]

            # find the causal drivers for each timestep, 
            # notes: 
            #  1) we didn't involve past information of leaf node. 
            #     Instead we used all past information for each leaf node. 
            #  2) if don't have any drivers, we don't save in tree dict
            driver_t_leaf = list(np.where(link_t_leaf >= self.sig_thres)[0])
            impact_t_leaf = impact_matrix[driver_t_leaf, -1, timestep]

            driver_t_leaf = [i for i in driver_t_leaf if i != self.num_features-1]

            if len(driver_t_leaf) != 0: # empty list equal to False
                self.causal_link_groups[str(timestep)] = driver_t_leaf
                self.causal_impact_groups[str(timestep)] = impact_t_leaf
            
    def prune_causal_link(self, inputs):
        for (timestep, causal_link) in self.causal_link_groups.items():
            _causal_link = []
            for causal_drivers in causal_link:
                r, p = pearsonr(inputs[:,-1], inputs[:,causal_drivers])
                if p <= 0.05: #and r>=0.3:
                    _causal_link.append(causal_drivers)
            
            self.causal_link_groups[timestep] = _causal_link

    def get_group_trees(self):
        """generate inputs for Causal LSTM for each single tree in group."""
        self.group_node_dict = {}
        self.group_num_child_nodes = {}
        self.group_input_idx = {} 
        self.group_child_state_idx = {}

        for (timestep, causal_link) in self.causal_link_groups.items():
            node_dict, num_child_nodes, input_idx, child_state_idx = \
                self._get_one_tree(causal_link)

            self.group_node_dict[timestep] = node_dict
            self.group_num_child_nodes[timestep] = num_child_nodes
            self.group_input_idx[timestep] = input_idx
            self.group_child_state_idx[timestep] = child_state_idx

    def _get_num_child_nodes(self, node_dict):
        """get number of child nodes for each node."""
        num_child_nodes = []
        
        for level in np.arange(self.depth, 0, -1):
            if level == self.depth:
                for node_level in node_dict[str(level)]:
                    for node in node_level:
                        num_child_nodes.append(0)
            else:
                for node_level in node_dict[str(level+1)]:
                    num_child_nodes.append(len(node_level))

        return num_child_nodes

    def _get_input_idx(self, node_dict):
        """get input index of node """
        input_idx = []
        
        for level in np.arange(self.depth,0,-1):
            for node_level in node_dict[str(level)]:
                for node in node_level:
                    input_idx.append([int(node)])   

        return input_idx

    def _get_child_state_idx(self, node_dict):
        """get state of child nodes of each node."""
        child_state_idx = []
        count = -1

        for level in np.arange(self.depth,0,-1):
            if level == self.depth:
                for node_level in node_dict[str(level)]:
                    for node in node_level:
                        child_state_idx.append([])
            else:
                for node_level in node_dict[str(level+1)]:
                    _child_state_idx = []

                    for node in node_level:
                        count += 1
                        _child_state_idx.append(count)
                    
                    child_state_idx.append(_child_state_idx)

        return child_state_idx

    def _get_one_tree(self, causal_link):
        """generate inputs for Causal LSTM for each single tree in group."""
        # generate feature dict for each level.
        node_dict = {}
        node_dict['1'] = [[self.num_features-1]]
        node_dict['2'] = [causal_link] # may change

        # generate inputs of CLSTM using node_dict
        num_child_nodes = self._get_num_child_nodes(node_dict)
        input_idx = self._get_input_idx(node_dict)
        child_state_idx = self._get_child_state_idx(node_dict)

        return node_dict, num_child_nodes, input_idx, child_state_idx