"""This file containts the data generation process 
of confounded MDP. For more details of the confounded
MDP please look at Appendix B of the paper. 

The main part of the code is borrowed from data generator:
https://github.com/clinicalml/gumbel-max-scm/blob/master/sepsisSimDiabetes/DataGenerator.py

File containts two classes
    confounded_matrix_mdp: simulates the confounded mdp with
        transition and reward matrix. Simular to MatrixMDP class 
        in https://github.com/clinicalml/gumbel-max-scm
    
    conf_data_generator: a wrapper around confounded_matrix_mdp
        that faciliates data generation prcess.
"""

import numpy as np, random
from .sepsisSimDiabetes.MDP import MDP
from .sepsisSimDiabetes.State import State
from .sepsisSimDiabetes.Action import Action
from tqdm import tqdm

class confounded_matrix_mdp(object):
    def __init__(self, tx_mat, r_mat, policy, t0_policy, value_fn, config):
        """__init__
        Implements an MDP with a transiton and reward matrix, such that the first step 
            policy can be different from the rest, and first step may be confounde.
        
        Parameters
        ----------
        tx_mat: np.array, float  [n_actions, n_states, n_states]
            Transition matrix of shape [n_actions, n_states, n_states]
        r_mat : np.array, float [n_actions, n_states, n_states]
            Reward matrix of shape [n_actions, n_states, n_states]
        policy : np.array, float [n_action, n_states]
            Probability distribution over next action given state 
            This policy is used from t=1 onward.
        t0_policy : np.array, float, [2, n_actions, n_states]
            Two policies, first to upweight if confounder is good
                second to upweight if confounder is bad with shape
                This policy is used at t=0
        value_fn : np.array, float, [n_states]
            value function of the optimal poilicy 
        config : dictionary containts the following
            max_horizon : int
                maximum number of timesteps in each simulation
            Gamma : flaot
                amount of confounding
            confonding_threshold : float
            p_diabetes : float 
                probability of a diabitic patinet

        Methods
        -------
        generate_random_state()
            generates a random state as initial state
            returns a State object
        reset()
            reset the MDP and return the initial state
        _choice()
            pick the next state based on confounding value, and
            returns the index of the next state
        step()
            takes a step based on policy and returns 
            next state, reward, if terminal
        _select_actions_unconfounded()
            select an action based on policy -- unconfounded
            returns action index
        _select_actions_confounded()
            select an action based on policy -- confounded
            returns action index
        select_actions()
            select the next action, either confounded or un confounded
            returns the action index
        """

        # size of the inputs
        assert tx_mat.ndim == 3, \
            "Transition matrix wrong dims ({} != 3)".format(tx_mat.ndim)
        assert r_mat.ndim == 3, \
            "Reward matrix wrong dims ({} != 3)".format(tx_mat.ndim)
        assert r_mat.shape == tx_mat.shape, \
            "Transition / Reward matricies not the same shape!"
        assert tx_mat.shape[-1] == tx_mat.shape[-2], \
            "Last two dims of Tx matrix should be equal to num of states"

        # Get the number of actions and states
        n_actions = tx_mat.shape[0]
        n_states = tx_mat.shape[1]

        self.Gamma = config['Gamma']
        self.confounding_threshold = config['confounding_threshold']
        self.max_horizon = config['max_horizon']
        self.p_diabetes = config['p_diabetes']

        self.n_actions = n_actions
        self.n_states = n_states
        self.tx_mat = tx_mat
        self.r_mat = r_mat

        self.current_state = None
        
        self.policy = policy
        self.t0_policy = t0_policy
        self.value_fn = value_fn
        

    def generate_random_state(self):
        """generates a random initial state
        with pre defined prior.

        Returns
        -------
        State : fully specified State object
        """
        # Note that we will condition on diabetic idx if provided
        diabetic_idx = np.random.binomial(1, self.p_diabetes)

        # hr and sys_bp w.p. [.25, .5, .25]
        hr_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        # percoxyg w.p. [.2, .8]
        percoxyg_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))

        if diabetic_idx == 0:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.01, .05, .15, .6, .19]))
        antibiotic_state = 0
        vaso_state = 0
        vent_state = 0

        state_categs = [hr_state, sysbp_state, percoxyg_state,
                glucose_state, antibiotic_state, vaso_state, vent_state]

        return State(state_categs=state_categs, diabetic_idx=diabetic_idx)

    def reset(self):
        """reset
        Reset the environment by 
            1. reseting the initial state
            2. confounding values (self.confounders)
            3. setting self.time = 0
        Returns
        -------
        self.current_state : int
            index of the current (initial) state
        """
        # Draw an initial state
        init_state = self.generate_random_state()
        while init_state.check_absorbing_state():
            init_state = self.generate_random_state()
        self.current_state = init_state.get_state_idx()
    
        # Draw unobserved confounders
        self.confounders = np.random.uniform(0, 1, size=self.max_horizon)
        self.time = 0

        return self.current_state

    def _choice(self, probs, u, idx):
        """_choice
        Picks the next state. The first state "i" 
        such that u > \sum{p(i)}, where p(i) is probability of state i
        and states are sorted based on R + \gamma V  

        Parameters
        ----------
        probs : np.array, float [n_states]
            probability of the next state for each state
        u : float
            confounder value
        idx : np.array, int [n_state]
            sorted indexes of states based on R(s,a) + \gamma V(s')
        
        Returns
        -------
        state : int
            index of the next state
        """
        cumulative = 0
        for i in range(self.n_states):
            state = idx[i]
            p = probs[state]
            cumulative += p
            if u < cumulative:
                return state
        return  idx[-1]

    def step(self, action):
        """step
        Take a step with the given action

        Patameters
        ----------
        action : int
            index of the action
        
        Returns
        -------
        self.current_state : int
            index of the next state
        reward : float
            reward of (state, action, next state)
        is_term : bool
            if self.current_state is a terminal state
        """
        assert action in range(self.n_actions), "Invalid action!"
        assert self.time < self.max_horizon, "Out of time horizon!"
        is_term = False

        next_prob = self.tx_mat[action, self.current_state,
                :].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"
        sorted_idx = np.argsort(self.value_fn +\
                                self.r_mat[action, self.current_state, :])
        next_state = self._choice(next_prob, self.confounders[0], sorted_idx)

        reward = self.r_mat[action, self.current_state, next_state]
        self.current_state = next_state

        # In this MDP, rewards are only received at the terminal state
        if reward != 0:
            is_term = True

        self.time += 1
        if self.time >= self.max_horizon:
            is_term = True

        return self.current_state, reward, is_term

    def _select_actions_unconfounded(self):
        """_select_actions_unconfounded
        selects the next action based on the policy, unconfounded

        Returns
        -------
        next_action : int
            index of the next action
        """
        next_prob = self.policy[:, self.current_state].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"

        next_action = np.random.choice(self.n_actions, size=1, p=next_prob)[0]

        return next_action

    def _select_actions_confounded(self):
        """_select_actions_confounded
        selects the next action based on the policy, confounded
        when the confounder is high, upweight the with antibiotics
        policy.

        Returns
        -------
        next_action : int
            index of the next action
        """
        if (np.sqrt(self.max_horizon) * (self.confounders[0] - 0.5)) > self.confounding_threshold:
            if np.random.uniform(0,1) <= np.sqrt(self.Gamma) / (1 + np.sqrt(self.Gamma)):
                policy = self.t0_policy[0,...]
            else:
                policy = self.t0_policy[1,...]
        else:
            if np.random.uniform(0,1) <= np.sqrt(self.Gamma) / (1 + np.sqrt(self.Gamma)):
                policy = self.t0_policy[1,...]
            else:
                policy = self.t0_policy[0,...]

        next_prob = policy[:, self.current_state].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"

        next_action = np.random.choice(self.n_actions, size=1, p=next_prob)[0]

        return next_action

    def select_actions(self):
        """select_actions
        at time = 0 calls the confounded action selection
        and unconfounded action selction otherwise.

        Returns
        -------
        next_action : int
            index of the next action
        """
        if self.time == 0:
            next_action = self._select_actions_confounded()
        else:
            next_action = self._select_actions_unconfounded()
        return next_action


class conf_data_generator(object):
    def __init__(self, transitions, policies, value_fn, config):
        
        """__init__
        wrapper around confounded_matrix_mdp to simulate trajectories from confounded MDP
        
        Parameters
        ----------
        transitions: tuple (tx, tr)
            tx : np.array, float [n_actions, n_states, n_states]
                transition matrix
            tr : np.array, float [n_actions, n_states, n_states]
                reward matrix
        policies : tuple (policy, t0_policy)
            policy : np.array, float [n_actions, n_states]
                probability distribution over next state given action, for t=1 onward
            t0_policy : np.array, float [2, n_actions, n_states]
                two policies, first to upweight if confounder is good
                second to upweight if confounder is bad
        value_fn : np.array, float [n_states]
             value function of the optimal policy
        config : dictionary containing:
            max_horizon : int
                maximum number of timesteps in each simulation
            Gamma : float
                amount of confounding
            confonding_threshold
            p_diabetes : float
                probability of a diabitic patinet
            discount : float
                MDP's discount factor

        Methods
        -------
        simulate(num_iters)
            simulates num_iters trajectories and rewards by confounded_MDP
            returns trajectories and discounted returns
        process_data()
            process output of confounded MDP to trajectories and returns
        calc_reward()
            calculates discounted return
        """

        tx_mat, r_mat = transitions
        policy, t0_policy = policies

        self.MDP = confounded_matrix_mdp(tx_mat, r_mat, policy, t0_policy, 
                                       value_fn, config)
        self.max_num_steps = config['max_horizon']
        self.config = config

    def simulate(self, num_iters, use_tqdm=False):
        """simulates 
        Simulates num_iters trajectories and rewards by confounded_MDP
        
        Parameters
        ----------
        num_iters : int
            number of simulations
        use_tqdm : bool
            if use tqdm while running
        
        Returns
        -------
        trajectories : np.array [num_iters, max_horizon, 5]
            [num_iters, max_horizon, 0] : timestep
            [num_iters, max_horizon, 1] : action taken, -1 default
            [num_iters, max_horizon, 2] : state index
            [num_iters, max_horizon, 3] : next state index
            [num_iters, max_horizon, 4] : reward
        returns : np.array [num_iters]
            discounted returns by config['discount'] discount factor
        
        """

        # Set the default value of states / actions to negative -1,
        # corresponding to None
        iter_states = np.ones((num_iters, self.max_num_steps+1, 1), dtype=int)*(-1)
        iter_actions = np.ones((num_iters, self.max_num_steps, 1), dtype=int)*(-1)
        iter_rewards = np.zeros((num_iters, self.max_num_steps, 1))

        for itr in tqdm(range(num_iters), disable=not(use_tqdm)):
            # MDP will generate the diabetes index as well
            state = self.MDP.reset()

            iter_states[itr, 0, 0] = state #self.initial_state
            for step in range(self.max_num_steps):
                action = self.MDP.select_actions()
                # Take the action
                state, reward, terminal = self.MDP.step(action)

                iter_actions[itr, step, 0] = action
                iter_states[itr, step+1, 0] = state
                iter_rewards[itr, step, 0] = reward

                if terminal :
                    iter_rewards[itr, step, 0] = reward
                    break
        trajectories, returns = self.process_data(
                                iter_states[..., 0], iter_actions[..., 0], 
                                iter_rewards[..., 0], num_iters)

        return trajectories, returns

    
    def process_data(self, states, actions, rewards, num_iters):
        """process_data 
        process states, actions and rewards into two arrays
        trajectories and rewards
        
        Parameters
        ----------
        states : np.array [num_iters, max_horizon]
            int, index of states
        actions : np.array [num_iters, max_horizon]
            int, index of actions            
        rewards : np.array [num_iters, max_horizon]
            float, index of actions            
        num_iters : int
            number of iterations
        
        Returns
        -------
        trajectories : np.array [num_iters, max_horizon, 5]
            [num_iters, max_horizon, 0] : timestep
            [num_iters, max_horizon, 1] : action taken, -1 default
            [num_iters, max_horizon, 2] : state index
            [num_iters, max_horizon, 3] : next state index
            [num_iters, max_horizon, 4] : reward
        disc_reward : np.array [num_iters]
            discounted returns by config['discount'] discount factor
        """

        disc_reward = np.zeros(num_iters)
        trajectories = np.zeros((num_iters, self.config['max_horizon'], 5))
        trajectories[:, :, 0] = np.arange(self.config['max_horizon'])  # Time Index
        trajectories[:, :, 1] = actions # actions
        trajectories[:, :, 2] = states[:, :-1]  # from_states
        trajectories[:, :, 3] = states[:, 1:]  # to_states
        trajectories[:, :, 4] = rewards # rewards
        disc_reward = self.calc_reward(trajectories, discount=self.config['discount'])

        return trajectories, disc_reward

    def calc_reward(self, trajectories, discount):
        """calc_reward
        calculates the discounted return

        Parameters
        ----------
        trajectories : np.array [num_iters, max_horizon, 5]
            output of process_data
        discout : float
            discount factor
        
        Returns
        -------
        discounted_reward : float [num_iters]
            discounted return for each trajectoru
        """
        # Column 0 is a time index, column 4 is the reward
        discounted_reward = (discount**trajectories[..., 0] * trajectories[..., 4])
        return discounted_reward.sum(axis=-1)  # Take the last axis