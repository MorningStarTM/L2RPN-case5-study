import numpy as np
import torch
import random
from itertools import product
from grid2op import Environment


class ActionConverter:
    def __init__(self, env:Environment) -> None:
        self.action_space = env.action_space
        self.env = env
        self.sub_mask = []
        self.init_sub_topo()
        self.init_action_converter()

    def init_sub_topo(self):
        self.subs = np.flatnonzero(self.action_space.sub_info)
        self.sub_to_topo_begin, self.sub_to_topo_end = [], [] # These lists will eventually store the starting and ending indices, respectively, for each actionable substation's topology data within the environment's overall topology information.
        idx = 0 # This variable will be used to keep track of the current position within the overall topology data
        
        for num_topo in self.action_space.sub_info: # The code can efficiently extract the relevant portion of the overall topology data that specifically applies to the given substation
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def init_action_converter(self):
        self.actions = [self.env.action_space({})]
        self.n_sub_actions = np.zeros(len(self.action_space.sub_info), dtype=int)
        for i, sub in enumerate(self.subs):
            
            # Generating Topology Actions
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub) # retrieves all possible topology actions for the current substation using the get_all_unitary_topologies_set method of the action_space object
            self.actions += topo_actions  # Appends the topology actions for the current substation to the actions list.
            self.n_sub_actions[i] = len(topo_actions) # Stores the number of topology actions for the current substation in the n_sub_actions array
            self.sub_mask.extend(range(self.sub_to_topo_begin[sub], self.sub_to_topo_end[sub])) # Extends the sub_mask list with indices corresponding to the topologies of the current substation.
        
        self.sub_pos = self.n_sub_actions.cumsum() 
        self.n = len(self.actions)

    def act(self, action:int):
        return self.actions[action]
    
    def action_idx(self, action):
        return self.actions.index(action)
    
    

class Converter:
    def __init__(self, env, case):
        self.env = env
        self.n_powerlines = self.env.n_line
        self.n_substations = self.env.n_sub
        self.total_bus_actions = 21 if case=="rte_case5_example" else 56
        self.n_actions = 132 if self.total_bus_actions==21 else 347
        self.one_hot_encoding_act_conv, self.env_act_dict_list = self.create_one_hot_converter()

    def create_one_hot_converter(self):
        """
        Creates two 2-d np.arrays used for conversion between grid2op action to one hot encoding action vector used by a neural network
        """
        one_hot_encoding_act_conv = []
        env_act_dict_list = []
        zero_act = np.zeros((self.n_powerlines+self.total_bus_actions,1))

        ## Add do nothing action vector (all zeroes)
        one_hot_encoding_act_conv.append(zero_act)
        env_act_dict_list.append({}) ## {} is the do nothing dictonary for actions in grid2op

        ## Powerline change actions
        for idx in range(self.n_powerlines):
            one_hot_encoding_act_conv_pwline = zero_act.copy()
            one_hot_encoding_act_conv_pwline[self.total_bus_actions+idx] = 1
            one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_pwline)
            env_act_dict_list.append({'change_line_status': [idx]}) ## {'change_line_status': [idx]} set an action of changing line status for lineid with id idx


        ## Bus change actions
        start_slice = 0
        for sub_station_id, nb_el in enumerate(self.env.action_space.sub_info):
            one_hot_encoding_act_conv_substation = zero_act.copy()

            possible_bus_actions = np.array(list(product('01', repeat=nb_el))).astype(int)
            for possible_bus_action in possible_bus_actions:
                if possible_bus_action.sum()>0: # Do not include no change action vector
                    one_hot_encoding_act_conv_substation[start_slice:(start_slice+nb_el)] = possible_bus_action.reshape(-1,1)
                    one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_substation.copy())
                    env_act_dict_list.append({"change_bus": {"substations_id": [(sub_station_id, possible_bus_action.astype(bool))]}})
            start_slice += nb_el

        one_hot_encoding_act_conv = np.array(one_hot_encoding_act_conv).reshape(len(one_hot_encoding_act_conv),self.n_powerlines+self.total_bus_actions)

        return one_hot_encoding_act_conv,env_act_dict_list

    def convert_env_act_to_one_hot_encoding_act(self,env_act):
        """
        Converts an grid2op action (in numpy format) to a one hot encoding vector
        """
        
        one_hot_encoding_act = np.zeros(len(self.one_hot_encoding_act_conv))
        env_act = env_act.reshape(-1,)
        action_idx = (self.one_hot_encoding_act_conv[:, None] == env_act).all(-1).any(-1)
        one_hot_encoding_act[action_idx] = 1
        return one_hot_encoding_act

    def convert_one_hot_encoding_act_to_env_act(self,one_hot_encoding_act):
        """
        Converts a one hot encoding action to a grid2op action
        """
        return self.env.action_space(self.env_act_dict_list[one_hot_encoding_act.argmax().item()])
    
    def int_to_onehot(self, n):
        v = [0] * self.n_actions
        v[n] = 1
        return np.array(v)

class Node:
    def __init__(self, env):
        self.env = env
        self.obs = env.reset()
        self.node_types = ['substation', 'load', 'generator', 'line']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_substation_data(self):
        return self.obs.time_before_cooldown_sub
    
    def extract_load_data(self):
        return self.obs.load_p, self.obs.load_q, self.obs.load_v , self.obs.load_theta
    
    def extract_gen_data(self):
        return self.obs.gen_p.tolist(), self.obs.gen_q.tolist(), self.obs.gen_v.tolist(), self.obs.gen_theta.tolist()
    
    def extract_line_data(self):
        return self.obs.p_or, self.obs.q_or, self.obs.v_or, self.obs.a_or, self.obs.theta_or, self.obs.p_ex, self.obs.q_ex, self.obs.v_ex, self.obs.a_ex, self.obs.theta_ex, self.obs.rho, self.obs.line_status, self.obs.time_before_cooldown_line, self.obs.time_next_maintenance, self.obs.duration_next_maintenance

    def create_data(self):
        # Extract data for each node type
        substation_data = np.array([self.extract_substation_data()]).T
        load_data = np.array(self.extract_load_data()).T
        gen_data = np.array(self.extract_gen_data()).T
        line_data = np.array(self.extract_line_data()).T

        max_length = len(substation_data[0]) + len(load_data[0]) + len(gen_data[0]) + len(line_data[0])


        # Pad feature arrays to match the maximum length
        sub_padd = np.pad(substation_data, ((0, 0), (0, max_length - len(substation_data[0]))), mode='constant')
        load_padd = np.pad(load_data, ((0, 0), (0, max_length - len(load_data[0]))), mode='constant')
        gen_padd = np.pad(gen_data, ((0, 0), (0, max_length - len(gen_data[0]))), mode='constant')
        line_padd = np.pad(line_data, ((0, 0), (0, max_length - len(line_data[0]))), mode='constant')

        # Combine padded feature arrays into a single array
        feature_data = np.concatenate((sub_padd, load_padd, gen_padd, line_padd), axis=0)

        # Return the combined feature array
        return feature_data, self.obs.connectivity_matrix()
    
    def convert_obs(self, obs):
        # Convert observation to tensor format
        obs_vect = obs.to_vect()
        obs_vect = torch.FloatTensor(obs_vect).unsqueeze(0)
        length = self.env.action_space.dim_topo

        # Initialize tensors for features and edges
        rho_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        p_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        danger_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        
        # Fill in feature tensors with observation data
        rho_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        rho_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        p_[..., self.env.action_space.gen_pos_topo_vect] = torch.tensor(obs.gen_p, device=self.device)
        p_[..., self.env.action_space.load_pos_topo_vect] = torch.tensor(obs.load_p, device=self.device)
        p_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.p_or, device=self.device)
        p_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.p_ex, device=self.device)
        danger_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float()
        danger_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float() 

        # Stack feature tensors along the third dimension
        state = torch.stack([p_, rho_, danger_], dim=2).to(self.device)

        # Convert adjacency matrix to edge tensor
        adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)
        adj_matrix = np.triu(adj.cpu(), k=1) + np.triu(adj.cpu(), k=1).T
        edges = np.argwhere(adj_matrix)
        edges = edges.T
        edges_tensor = torch.tensor(edges, dtype=torch.long).to(self.device)
        
        # Pad edge tensor to fixed length
        max_edge_length = length * length  # Assuming adjacency matrix is square
        if edges_tensor.size(1) < max_edge_length:
            padding_length = max_edge_length - edges_tensor.size(1)
            padding = torch.zeros(2, padding_length, dtype=torch.long, device=self.device)
            edges_tensor = torch.cat([edges_tensor, padding], dim=1)

        return state, edges_tensor

    
    def standard_normalize(self, obs):
        obs_vect = obs.to_vect()
        mean_obs = np.mean(obs_vect, axis=0)
        std_obs = np.std(obs_vect, axis=0)
        normalized_obs = (obs_vect - mean_obs) / std_obs
        return normalized_obs




