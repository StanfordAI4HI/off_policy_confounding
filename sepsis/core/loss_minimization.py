"""Performing loss minimization and computing lower bound on
an evaluation policy with an unobserved confounding

Off-policy Policy Evaluation Under Unobserved Confounding
Ramtin keramati, Steve Yadlowsky, Hongseok Namkoong, Emma Brunskill
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

class loss_minimization(object):
    """Class to perform loss minimization and evaluates lowerbound
        on an evaluation policy with an unobserved confounding. This
        class assumes confounding happens at the first time step
    
    Attributes
    ----------
    Gamma : float
		amount of confounding injected in the simulation
    bootstrap : bool
        if perform bootstrapping
    n_bootrstrp : int 
        number of bootstraps
    nA : int
        number of actions
    nS : int
        number of states
    lr : float
        learning rate, used for optimizer
    batch_size : int
        batch size for neural net training, if None whole data
    pi_e_t0 : np.array, float [n_actions, n_states]
        evaluation policy at time step 0
    pi_e_t1 : np.array, float [n_actions, n_states]
        evaluation policy at time step 1 and afterward
    data : np.array, float [num_samples, max_horizion, 5]
        [num_samples, max_horizon, 0] : timestep
        [num_samples, max_horizon, 1] : action taken, -1 default
        [num_samples, max_horizon, 2] : state index
        [num_samples, max_horizon, 3] : next state index
        [num_samples, max_horizon, 4] : reward
    
    Methods
    -------
    create_placeholders()
        creates tensorflow placeholders
    create_network()
        creates neural net for estimating k(s, a)
    create_loss()
        creates loss operator for optimizer
    create_trainop()
        creates training operator for tensorflow
    setup_data(data)
        process the input (bootstrap) data 
    learn_behaviour_policies(trajectories)
        learns two behaviour policy from input data
    get_feed_dict(), get_eval_feed_dict()
        generates feed dict for tensorflow for training and evaluation    
    compute_lowerbound()
        computes the lower boudn given kappa, policy
    compute_trivial_lowerbound() 
        computes the trivial lowerbound for evaluation policy
    train()
        train the neural net
    run()
        run training, evaluating and computing lower bound
    """
    def __init__(self, config, data, evaluation_policies, scope):
        """__init__, sets up the configuration
        Parameters
        ----------
        config: dictionary containing class attributes -- see above 
        data : dictionary with keys
            samps : np.array, float [num_samples, horizon, 5] -- see self.data
            returns : np.array, float [num_samples, 1]
                discounted return of each trajecotry
        evaluation_policies : (pi_e_t0, pi_e_t1)
            pie_e_t0 : np.array, float [n_actions, n_states]
                evaluation policy at time step 0
            pie_e_t1 : np.array, float [n_actions, n_states]
                evaluation policy at time step 1 and onward
        scope : string
            defining the variable scope
        """
        
        self.Gamma = config['Gamma']
        self.lr = config['lr']
        self.epoch = config['epoch']
        self.nS = config['nS']
        self.nA = config['nA']
        self.bootstrap = config['bootstrap']
        if self.bootstrap:
            assert 'n_bootstrap' in config.keys(), "bootstrap mode is on, without setting the number of bootstraps"
            self.n_bootstrap = config['n_bootstrap']
        self.data = data
        self.num_data = self.data['samps'].shape[0]
        if 'batch_size' in config.keys():
            self.batch_size = config['batch_size']
        else:
            self.batch_size = self.num_data
        self.num_iteration = int(self.num_data * self.epoch/ self.batch_size)

        # setting evaluation policy
        self.pi_e_t0, self.pi_e_t1 = evaluation_policies

        # create tensorflow graph
        with tf.variable_scope(scope):
            self.create_placeholders()
            self.create_network()
            self.create_loss()
            self.create_train_op()
            # Session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
    
    def update_gamma(self, Gamma):
        """update confonding value"""
        self.Gamma = Gamma        

    def run(self, upper_bound=False, use_tqdm=False):
        """Runs the class and returns a tuple, lowerbound and trivial lowerbound
            Function runs by the following steps:
            1. Sets up the data and learn behaviour policies (with or without bootstrap)
            2. Train the network for each state, acion paier (tabular)
            3. Computes the lowerbound  and the trivial lower bound given self.kappa(s, a)

        Parameters
        ---------
        use_tqdm : bool
            if use tqdm
        upper_bound : bool
            if true calculates upper bound (- lowerbound of negative return)

        Returns
        -------
        lowerbound : float
            if bootstrap:
                np.array [n_boottrap]
            else:
                float
        trivial_lower_bound : float
            if bootstrap:
                np.array [n_boottrap]
            else:
                float
        loss : np.array float
            loss value while trainning
            if bootstrap:
                np.array [n_boottrap, num_iterations]
            else:
                np.array [num_iterations]
        """
        if upper_bound:
            sign = -1.0 
        else:
            sign = 1.0

        if self.bootstrap:
            result, result_naive = np.zeros(self.n_bootstrap), np.zeros(self.n_bootstrap)
            loss = np.zeros((self.n_bootstrap, self.num_iteration))
            # for number of bootstrap
            for i in tqdm(range(self.n_bootstrap), disable=not use_tqdm):
                idxs = np.random.choice(np.arange(self.num_data),
                                        size=self.num_data, replace=True)
                # generate bootstrap data dictionary
                bootstrap_data = {'samps': self.data['samps'][idxs, :, :], 
                                  'returns': sign * self.data['returns'][idxs]}
                # set the data, and learn the behaviour policy
                self.setup_data(bootstrap_data)
                # loss minimization:
                loss[i, :] = self.train()
                # evaluation
                result[i], result_naive[i] = self.compute_lowerbound() * sign, self.compute_trivial_lowerbound() * sign
            return result, result_naive, loss
        else:
            data = {'samps': self.data['samps'], 
                    'returns': sign * self.data['returns']}
            # set the data, and learn the behaviour policy
            self.setup_data(data)
            # loss minimization:
            loss = self.train()
            # evaluation
            return self.compute_lowerbound() * sign, self.compute_trivial_lowerbound() * sign, loss
                
    def setup_data(self, data):
        """setup_data
            seprates states and action for more readibility
            and learns the behaviour poliocy
        """
        self.states = data['samps'][:, :, 2]
        self.actions = data['samps'][:, :, 1]
        self.returns = np.expand_dims(data['returns'], axis=-1)
        self.learn_behaviour_policies(data['samps'])
        # set the pointer of the data reader to 0
        self.pointer = 0 

    def learn_behaviour_policies(self, trajectories):
        """ learns two different policies, one before first k step, 
            and one after first k step
        Parameters
        ----------
        trajectories : output of conf_data_generator.simulate()
        """
        self.pi_b_t0 = self._learn_t0_policy(trajectories)
        self.pi_b_t1 = self._learn_t1_policy(trajectories) 

    def _learn_t0_policy(self, obs):
        """learn tabular policy for first time step"""
        policy = np.zeros((self.nA, self.nS))
        for sample in range(obs.shape[0]):
            for step in range(obs.shape[1]):
                s = int(obs[sample, step, 2])
                a = int(obs[sample, step, 1])
                # only for t0
                if a==-1 or step > 1:
                    break
                policy[a, s] += 1
        nonzero = policy.sum(axis=0) > 0
        policy[:, nonzero] /= policy[:, nonzero].sum(axis=0, keepdims=True)
        return policy

    def _learn_t1_policy(self, obs):
        """learn tabular policy for after first time step"""
        policy = np.zeros((self.nA, self.nS))
        for sample in range(obs.shape[0]):
            # after t0
            for step in range(1, obs.shape[1]):
                s = int(obs[sample, step, 2])
                a = int(obs[sample, step, 1])
                if a==-1:
                    break
                policy[a, s] += 1
        nonzero = policy.sum(axis=0) > 0
        policy[:, nonzero] /= policy[:, nonzero].sum(axis=0, keepdims=True)
        return policy

    def create_placeholders(self):
        """ creates tf placeholders"""
        self.pl_p_b_t0 = tf.placeholder(dtype=tf.float32, shape = [None,1], name='behaviour_policy_t0')
        self.pl_p_e_t0 = tf.placeholder(dtype=tf.float32, shape = [None,1], name='evaluation_policy_t0')
        self.pl_p_e_t1 = tf.placeholder(dtype=tf.float32, shape = [None,1], name='eval_policy_prod')
        self.pl_p_b_t1 = tf.placeholder(dtype=tf.float32, shape = [None,1], name='behaviour_policy_prod')
        self.pl_Gamma = tf.placeholder(dtype=tf.float32, name='Gamma')
        self.pl_reward = tf.placeholder(dtype=tf.float32, shape = [None,1], name='total_return')
        self.pl_state = tf.placeholder(dtype=tf.int32, shape = [None, 1], name='state')
        self.pl_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.nA], name='indicator_func')

    def create_network(self):
        """create_network
        tf.Variable for tabular represetation of self.kappa(x,a)"""
        # store a value for each variable theta(x,a)
        self.kappa = tf.Variable(initial_value=tf.zeros(shape=(self.nS, self.nA)), 
                                 trainable=True, name='kappa', dtype=tf.float32)

    def create_loss(self):
        """Creates loss"""
        self.loss = 0
        # summation over all actions
        for a in range(self.nA):
            argument = tf.gather(self.kappa, tf.squeeze(self.pl_state), axis=0)
            argument = tf.gather(argument, a, axis=1)
            argument = tf.squeeze(self.pl_p_e_t1/self.pl_p_b_t1 * self.pl_reward) - argument

            kappa_1 = tf.maximum(argument, 0)
            kappa_2 = -1.0 * tf.minimum(argument, 0)
            
            self.loss += tf.reduce_mean(0.5 * tf.gather(self.pl_mask, a, axis=1)* \
                        (self.pl_Gamma * kappa_2**2 + kappa_1**2))

    def create_train_op(self):
        """Create train operator"""
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def get_eval_feed_dict(self):
        """ Get feed_dict for evaluation"""
        states = self.states[:, :].astype(int)
        actions = self.actions[:, :].astype(int)
        returns = self.returns[:, 0]
        returns = np.expand_dims(returns, axis=-1)

        terminated_idx = actions[:, 1:] == -1
        # indicator functions for a=A
        mask = np.zeros((states.shape[0], self.nA))
        for a in range(self.nA):
            mask[:, a] = (actions[:, 0] == a) * 1.0
        # evaluation policy
        p_e_t1 = self.pi_e_t1[actions[:, 1:], states[:, 1:]]
        p_b_t1 = self.pi_b_t1[actions[:, 1:], states[:, 1:]]
        # behaviour policy
        p_e_t1[terminated_idx] = 1
        p_b_t1[terminated_idx] = 1
        
        p_e_t1 = p_e_t1.prod(axis=1)
        p_b_t1 = p_b_t1.prod(axis=1)
        p_b_t0 = self.pi_b_t0[actions[:, 0], states[:, 0]]
        p_e_t0 = self.pi_e_t0[actions[:, 0], states[:, 0]]

        return p_e_t0, returns, p_e_t1, p_b_t1, p_b_t0
    
    def get_feed_dict(self):
        """ Get feed_dict for train"""
        if self.pointer + self.batch_size  > self.num_data:
            first_part = self.num_data - self.pointer
            second_part = self.batch_size - first_part
            states = np.concatenate([self.states[self.pointer:-1, :, np.newaxis],
                                    self.states[:second_part, :, np.newaxis]], axis=0).astype(int)
            actions = np.concatenate([self.actions[self.pointer:-1, :, np.newaxis],
                        self.actions[:second_part, :, np.newaxis]], axis=0).astype(int)
            returns = np.concatenate([self.returns[self.pointer:-1, 0, np.newaxis],
                        self.returns[:second_part, 0, np.newaxis]], axis=0)
            self.pointer = second_part
        else: 
            end = self.pointer + self.batch_size
            states = self.states[self.pointer: end, :].astype(int)
            actions = self.actions[self.pointer: end, :].astype(int)
            returns = self.returns[self.pointer: end, 0, np.newaxis]
            self.pointer += self.batch_size
            if self.pointer >= self.num_data: self.pointer = 0 # reset the pointer

        terminated_idx = actions[:, 1:] == -1

        mask = np.zeros((self.batch_size, self.nA))
        for a in range(self.nA):
            mask[:, a] = (actions[:, 0] == a) * 1

        p_e_t1 = self.pi_e_t1[actions[:, 1:], states[:, 1:]]
        p_b_t1 = self.pi_b_t1[actions[:, 1:], states[:, 1:]]

        p_e_t1[terminated_idx] = 1
        p_b_t1[terminated_idx] = 1

        p_e_t1 = p_e_t1.prod(axis=1)
        p_b_t1 = p_b_t1.prod(axis=1)

        p_b_t0 = self.pi_b_t0[actions[:, 0], states[:, 0]]

        first_state = np.expand_dims(states[:, 0], axis=-1)
        feed_dict = {self.pl_p_b_t0: p_b_t0[:, np.newaxis], self.pl_p_b_t1: p_b_t1[:, np.newaxis],
                     self.pl_p_e_t1: p_e_t1[:, np.newaxis], self.pl_mask:mask,
                     self.pl_state: first_state, self.pl_reward:returns,
                     self.pl_Gamma: self.Gamma}
        return feed_dict 

    def train(self):
        """train neural net
        Returns
        -------
        loss_value : float [num_iteration]
            loss value for each iteration
        """
        loss_value = np.zeros(self.num_iteration)

        for it in range(self.num_iteration):
            feed_dict = self.get_feed_dict()
            loss_value[it], _ = self.sess.run([self.loss, self.train_op],
                                                    feed_dict=feed_dict)
        return loss_value

    def compute_lowerbound(self):
        """Compute the lower bound,
        Returns
        -------
        lower_bound : float
            computed lower bound
        """
        p_e_t0, returns, p_e_t1, p_b_t1, p_b_t0 = self.get_eval_feed_dict()

        kappa = self.sess.run(self.kappa) # evluated theta for all (s,a): caling this once is faster

        lower_bound = 0
        for a in range(self.nA):
            eta = kappa[self.states[:, 0].astype(int), a]
            lower_bound += eta * (1-self.pi_b_t0[a, self.states[:, 0].astype(int)]) *\
                            self.pi_e_t0[a, self.states[:, 0].astype(int)]
        lower_bound = np.mean(lower_bound) + np.mean(p_e_t0 * p_e_t1/p_b_t1 * np.squeeze(returns))

        return lower_bound
    
    def compute_trivial_lowerbound(self):
        """Compute the trivial lower bound,
        Returns
        -------
        lower_bound : float
            computed trivial lower bound
        """
        # evaluation policy
        p_e = self.pi_e_t1[self.actions[:,:].astype(int), 
                           self.states[:,:].astype(int)]
        p_e[:, 0] = self.pi_e_t0[self.actions[:,0].astype(int), 
                                self.states[:,0].astype(int)]
        # behaviour policy
        p_b = self.pi_b_t1[self.actions[:,:].astype(int), 
                           self.states[:,:].astype(int)]
        p_b[:, 0] = self.pi_b_t0[self.actions[:,0].astype(int), 
                                self.states[:,0].astype(int)]       
        # adjust for horizon 
        terminated_idx = self.actions == -1
        p_b[terminated_idx] = 1
        p_e[terminated_idx] = 1

        rho_pos = (p_e / p_b).prod(axis=1) /self.Gamma
        rho_neg = (p_e / p_b).prod(axis=1) *self.Gamma
        idx_pos = self.returns[:, 0] >= 0
        idx_neg = self.returns[:, 0] < 0
        # is estimate
        is_pos = np.sum((rho_pos[idx_pos]) * self.returns[idx_pos, 0])
        is_neg = np.sum((rho_neg[idx_neg]) * self.returns[idx_neg, 0])

        lower_bound = (is_pos + is_neg)/self.num_data

        return lower_bound