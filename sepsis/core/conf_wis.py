"""This file containts conf_wis class that is used to
estimate weighted importance sampling of confounded MDP
"""
import numpy as np
from tqdm import tqdm

class conf_wis(object):
    def __init__(self, trajectories, returns, k, config):
        """Computing weighted importance sampling estimate
        for a confounded matrix MDP. This class assumes there 
        are two policies acting one before time step k, and one after.
        This class assumes tabular representation of state, action

        Parameters
        ----------
        trajectories : np.array, float [None, max_horizon, 5]
            [None, max_horizon, 0] : timestep
            [None, max_horizon, 1] : action taken, -1 default
            [None, max_horizon, 2] : state index
            [None, max_horizon, 3] : next state index
            [None, max_horizon, 4] : reward
        returns : np.array, float [None]
            discounted returns computed for each trajectory
        k : int
            after step k action selection is unconfounded
        config : dictionary containing
            nS : int
                number of states
            nA : int
                number of actions
            n_bootstrap : int
                number of bootstrap samples

        Methods
        -------
        compute()
            computes the WIS estimate for n_bootstrap
        _compute_wis()
            computes wis for one set of trajectories and returns
        _learn_split_policies()
            learn two behaviour policies, before k and after k
            _learn_first_k_step_policy()
            _learn_after_k_step_policy()
        """
        self.trajectories = trajectories
        self.returns = returns
        self.k = k
        self.config = config

    def compute(self, evaluation_policy, use_tqdm=True):
        '''compute
        computes the wis estimate for evaluation policy
        n_bootstrap times. This function learns the behaviour
        policies seprately for each bootstrap samples

        Parameters
        ----------
        evaluation_policy : (e_t0_policy, e_policy)
             tuple of two policies each [n_actions, n_states]
            - e_t0_policy : evaluation policy at step 0
            - e_policy : evaluation policy after step 1
        use_tqdm : bool
            if use tqdm
        Returns
        -------
        wis_estimate : np.array, float [n_bootstrap]
            WIS estimates
        '''
        wis_estimate = np.zeros(self.config['n_bootstrap'])
        for i in tqdm(range(self.config['n_bootstrap']), disable=not use_tqdm):
            # bootstrap indexes:
            idxs = np.random.choice(np.arange(self.trajectories.shape[0]),
                        size=self.trajectories.shape[0], replace=True)
            obs = self.trajectories[idxs, :]
            returns = self.returns[idxs]
            # learn the behaviour policy
            b_t0_policy, b_policy = self._learn_split_policies(obs)
            wis_estimate[i] = self._compute_wis(trajectories=obs, 
                        returns=returns, 
                        behaviour_policy = (b_t0_policy, b_policy), 
                        evaluation_policy=evaluation_policy)
        return wis_estimate

    def _compute_wis(self, trajectories, returns, behaviour_policy, evaluation_policy):
        """compute_wis: adopted from David's paper
        Weighted Importance Sampling for Off Policy Evaluation

        Parameters
        ----------
        trajectories : np.array, float [None, max_horizon, 5]
            see __init__
        returns : np.array, float [None]
            see __init__
        behaviour_policy : (b_t0_policy, b_policy)
            b_t0_policy : np.array, float [n_actions, n_states]
                behaviour policy at step 0 
            b_policy : np.array, float [n_actions, n_states]
                behaviour policy after step 1
        evaluation_policy : (e_t0_policy, e_policy)
            e_t0_policy : np.array, float [n_actions, n_states]
                evaluation policy at step 0
            e_policy : np.array, float [n_actions, n_states]
                evaluation policy after step 1

        Returns
        -------
        wis_est : float
            weighted importance sampling estiamte
        """
        b_t0_policy, b_policy = behaviour_policy
        e_t0_policy, e_policy = evaluation_policy

        assert returns.ndim == 1 # check 1D array

        obs_actions = trajectories[..., 1].astype(int)
        obs_states = trajectories[..., 2].astype(int)

        # Evluation policy importance weights:
        # after first step
        p_eval = e_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        
        # first step
        p_first_step = e_t0_policy[obs_actions[:,0],
                                    obs_states[:,0]]
        if len(p_first_step.shape) == 1:
            p_first_step = np.expand_dims(p_first_step, axis=-1)

        p_eval = np.concatenate([p_first_step, p_eval], axis=-1)

        # Behaviour policy importance weights:
        p_first_step = np.expand_dims(b_t0_policy[obs_actions[:,0], 
                                                obs_states[:,0]], -1)
        p_behaviour = b_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        p_behaviour = np.concatenate([p_first_step, p_behaviour], axis=-1)

        # Deal with variable length sequences by setting ratio to 1
        terminated_idx = obs_actions == -1
        p_behaviour[terminated_idx] = 1
        p_eval[terminated_idx] = 1

        assert np.all(p_behaviour > 0), "Some actions had zero prob under p_obs, WIS fails"

        cum_ir = (p_eval / p_behaviour).prod(axis=1)
        wis_idx = (cum_ir > 0)

        if wis_idx.sum() == 0:
            print("Found zero matching WIS samples, continuing")
            return np.nan

        wis = (cum_ir) * returns

        wis_est = wis.mean()
        return wis_est

    def _learn_split_policies(self, obs):
        """
        learns two different policies, one before first k step, 
            and one after first k step
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5] 

        Returns
        -------
        (before_k, after_k) : tuple of policies
            before_k : np.array, floart [n_actions, n_states]
                behaviour policy before step k
            after_k : np.array, floart [n_actions, n_states]
                behaviour policy after step k
        """
        before_k = self._learn_first_k_step_policy(obs)
        after_k = self._learn_after_k_step_policy(obs) 
        return (before_k, after_k)  

    def _learn_first_k_step_policy(self, obs):
        """leanr policy before after step k
        
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5]
        
        Returns
        -------
        policy : np.array, float [n_actions, n_states]
            learned policy
        """
        policy = np.zeros((self.config['nA'], self.config['nS']))
        for sample in range(obs.shape[0]):
            for step in range(obs.shape[1]):
                s = int(obs[sample, step, 2])
                a = int(obs[sample, step, 1])
                if a==-1 or step > self.k:
                    break
                policy[a, s] += 1
        nonzero = policy.sum(axis=0) > 0
        policy[:, nonzero] /= policy[:, nonzero].sum(axis=0, keepdims=True)
        return policy

    def _learn_after_k_step_policy(self, obs):
        """leanr policy after after step k
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5]
        
        Returns
        -------
        policy : np.array, float [n_actions, n_states]
            learned policy
        """
        policy = np.zeros((self.config['nA'], self.config['nS']))
        for sample in range(obs.shape[0]):
            for step in range(self.k, obs.shape[1]):
                s = int(obs[sample, step, 2])
                a = int(obs[sample, step, 1])
                if a==-1:
                    break
                policy[a, s] += 1
        nonzero = policy.sum(axis=0) > 0
        policy[:, nonzero] /= policy[:, nonzero].sum(axis=0, keepdims=True)
        return policy