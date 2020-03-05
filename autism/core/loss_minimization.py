"""Performing loss minimization and computing lower bound on
an evaluation policy with an unobserved confounding

Off-policy Policy Evaluation Under Unobserved Confounding
Ramtin keramati, Steve Yadlowsky, Hongseok Namkoong, Emma Brunskill
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from sklearn.linear_model import LogisticRegression

class LowerBound(object):
    """Class to perform loss minimization and evaluates lowerbound
    on an evaluation policy with an unobserved confounding.

    Attributes
    ----------
    Gamma : float
		amount of confounding injected in the simulation
    action_set : [int]
        list of integer consisting of all possible actions at the 
        confounding step
    lr : float
        learning rate, used for optimizer
    batch_size : int
        batch size for neural net training
    evaluation_policy : (int, int)
        tuple with evaluation policy (A1, A2)
    
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
    setup_data(evaluation)
        process the input data
    learn_behaviour_policy()
        learns the behaviour policy from input data with logistic regression
    get_feed_dict(), get_eval_feed_dict()
        generates feed dict for tensorflow for training and evaluation
    compute_kappa()
        computes k(s, a) for each state action pair
    get_behaviour_policy_t0(x, a)
        return probability of action :param a: at time step 0 for covariates x
    get_behaviour_policy_t1(x, a)
        return probability of action :param a: at time step 1 for covariates x      
    compute_lowerbound()
        computes the lower boudn given kappa, policy
    compute_trivial_lowerbound() 
        computes the trivial lowerbound for evaluation policy
    train()
        train the neural net
    run()
        run training, evaluating and computing lower bound

    NOTE: How to use, refer to autism.ipynb
        after initilization, __init__(config, train_data, eval_data, scope)
        call run() that does the followin steps -- see run()
    """
    def __init__(self, config, train_data, eval_data, scope):
        """__init__, sets up the configuration

        Parameters
        ----------
        config: dictionary containing
            Gamma : float
                amount of confounding at the second time step
            action_set : [int]
                List of possible action at confounding step
                evaluation action in {0, 1, 2}, corresponding to
                A2 having values {-1, 0, 1}
            batch_size: int 
                batch size for training if None: batch_size = size of data
            lr : double
                learning rate for Adam Optimzier
            epoch : int
                number of training epochs
            evalution_policy : (int, int)
                tuple of A1, A2 as evaluation policy
            slow_responder : bool
                if yes, only evaluates the policy on slow responders R = 0

        train_data : float np.array num_train_data x (num_covaraites + num_outputs)
            this is the output of DataGen class -- see data_gen.py
        eval_data : float np.array num_eval_data x (num_covaraites + num_outputs)
            same as train_data, but used for evaluation
        scope : str
            tensorflow variable scope
        """

        self.slow_responder = config['slow_responder']
        self.num_covaraites = 10
        self.scale_reward = np.max(train_data[:, -1]) #normalizing Y for input to network
        self.setup_data(train_data, evaluation=False)
        self.setup_data(eval_data, evaluation=True)
        self.pointer = 0 # pointer for reading data from train data

        self.Gamma = config['Gamma']
        self.action_set = config['action_set']
        self.lr = config['lr']
        self.epoch = config['epoch']
        self.evaluation_policy = config['evaluation_policy']
        if config['batch_size'] is not None:
            self.batchsize = config['batch_size']
        else:
            self.batchsize = self.train_data.shape[0]
        

        self.num_data = self.train_data.shape[0]
        self.num_iteration = int(self.epoch * self.num_data/self.batchsize)
        self.kappa = np.zeros((self.eval_data.shape[0], len(self.action_set)))

        # setup tensorflow graph
        with tf.variable_scope(scope):
            self.create_placeholders()
            self.create_network()
            self.create_loss()
            self.create_trainop()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        # Learn behaviour policy
        self.learn_behaviour_policy() # with the train data

        self.save_dir = './saved_models/'

    def run(self, save_models=False, use_tqdm=False):
        """ Runs the class and returns a tuple, lowerbound and trivial lowerbound
        Function runs by the following steps:
            1. Train the network for each action (possible at the confunding step)
            2. Computes k(s, a) for each state action pair in the data and save it 
                in self.kappa
            3. Computes the lowerbound given self.kappa

        Parameters
        ----------
        save_models : bool
            if save the tf model for each action before reseting for the new action
        use_tqdm : bool
            if use tqdm for inner loop training
        plot_loss : bool
            if plot the loss curve

        Returns
        -------
        lb : (float, float)
            tuple of (lowerbound, trivial lower bound)
        loss_values : np.array [num_iteration, len(action_set)]
            training loss value
        """
        loss_values = np.zeros((self.num_iteration, len(self.action_set)))
        for idx, action in enumerate(self.action_set):
            self.set_action(action, reset=True)
            loss_values[:, idx] = self.train(use_tqdm=use_tqdm, 
                            desc='{} out of {}'.format(idx+1, len(self.action_set)))
            if save_models:
                self.saver.save(self.sess, self.save_dir + 'action_%d'%(action))
            self.kappa[:, action] = self.compute_kappa()[:, 0]

        lb = self.compute_lowerbound()
        lb_trivial = self.compute_trivial_lowerbound()

        return (lb, lb_trivial), loss_values

    def set_action(self, action, reset=True):
        """ Set the training action
        Parameters
        ----------
        action : int
            training action
        reset : bool
            if True re-initialize the tf graph
        """
        self.A = action
        if reset:
            self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self):
        """ creates tf placeholders"""
        self.p_b_t1 = tf.placeholder(dtype=tf.float32, 
                      shape = [None,1], name='behaviour_policy_t1')
        self.mask = tf.placeholder(dtype=tf.float32, 
                      shape = [None,1], name='mask') # mask for A2 = training action (self.A)
        self.reward = tf.placeholder(dtype=tf.float32, 
                      shape = [None,1], name='total_return')
        self.state = tf.placeholder(dtype=tf.float32, 
                      shape = [None, self.num_covaraites], name='state') # (S1, S2, R1)

    def create_network(self):
        """ Build tf network"""
        X = tf.layers.dense(self.state, 128, activation=tf.nn.relu,
                         kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_1')
        X = tf.layers.dense(self.state, 128, activation=tf.nn.relu,
                         kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_2')
        X = tf.layers.dense(self.state, 128, activation=tf.nn.relu,
                         kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_3')
        X = tf.layers.dense(self.state, 64, activation=tf.nn.relu,
                         kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_4')
        self.out = tf.layers.dense(X, 1, activation=None,
                         kernel_initializer=tf.keras.initializers.glorot_normal(), name='Output_layer')

    def create_loss(self):
        """Creates loss"""
        kappa_1 = tf.maximum(self.reward - self.out, 0)
        kappa_2 = -1.0 * tf.minimum(self.reward - self.out, 0)
        self.loss = tf.reduce_mean(0.5 * (self.Gamma * kappa_2**2 + kappa_1**2) *\
                             self.mask/self.p_b_t1)

    def create_trainop(self):
        """Create train operator"""
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def setup_data(self, data, evaluation=False):
        """ Prepare training and evaluation data
        The structure of the processed data is
        processed_data : [num_data x 12] np.array float
            processed_data[0: 7] = age, gender, Iaamerican
                Icaucasian, Ihispanic, Iasian, R
            processed_data[7] = Y12 (outcome at the first visit)
            processed_data[8, 10] = A1 (in {0, 1} = {-1, 1}), A2 (in {0, 1, 2}={-1, 0, 1})
            processed_data[9] = A1 x R (interaction term)
            processed_data[11] = outcome, Y36 (outcome at the end)

            network input = processed_data[0: 11] = S1 + A1 + A1xR
            network label = processed_data[11]
        
        Parameters
        ----------    
        data : float np.array num_train_data x (num_covaraites + num_outputs)
                this is the output of DataGen class -- see data_gen.py
        evaluation : bool
            whether to set the evaluation or train data
        """

        if self.slow_responder:
            data = data[data[:, 7]==0, :] # filter for R=0

        processed_data = np.zeros((data.shape[0], self.num_covaraites + 2))
        processed_data[:, 0:7] = data[:, 1:8] # S1 + R
        processed_data[:, 7] = data[:, 11]/self.scale_reward # S1 + R + Y12
        processed_data[:, 8] = (data[:, 8] != -1).astype(int) # A1 \in 0, 1
        processed_data[:, 9] = processed_data[:, 8] * processed_data[:, 6]# A1 x R
        processed_data[:, 10] = data[:, 9] + 1 # A2 \in 0, 1, 2
        processed_data[:, 11] = data[:, -1]/self.scale_reward # Y36 # return

        if evaluation:
            self.eval_data = processed_data
        else:
            self.train_data = processed_data

    def learn_behaviour_policy(self):
        """Learning behaviour policy from data with logistic regression.
        First time step: self.behaviour_policy_0, 
            input self.train_data[:, 0:6]:  age, gender, aamerican
                    caucasian, hispanic, asian
            output A1: self.train_data[:, 8]

        Second time step: self.behaviour_policy_1,
            input self.train_data[:, 0:10]:  age, gender, aamerican
                    caucasian, hispanic, asian, R, A1, RxA1
            output A2: self.train_data[:, 10]
        """
        self.behaviour_policy_0 = LogisticRegression(random_state=0, 
                    solver='newton-cg')
        X, y = self.train_data[:, 0:6], self.train_data[:, 8]
        self.behaviour_policy_0.fit(X, y)
        
        self.behaviour_policy_1 = LogisticRegression(random_state=0, 
                    solver='newton-cg')
        X, y = self.train_data[:, [0,1 ,2 ,3 ,4 ,5 ,6, 8, 9]], self.train_data[:, 10]
        self.behaviour_policy_1.fit(X, y)

    def get_eval_feed_dict(self):
        """ Get feed_dict for evaluation"""
        return {self.state: self.eval_data[:, 0:10]}

    def get_feed_dict(self):
        """Get feed_dcit for training"""
        if self.pointer + self.batchsize  > self.num_data:
            first_part = self.num_data - self.pointer
            second_part = self.batchsize - first_part
            batch = np.concatenate([self.train_data[self.pointer:-1, :],
                                    self.train_data[:second_part, :]], axis=0)
            self.pointer = second_part
        else: 
            end = self.pointer+self.batchsize
            batch = self.train_data[self.pointer: end, :]
            self.pointer += self.batchsize
            if self.pointer >= self.num_data: self.pointer = 0 # reset the pointer

        X = batch[:, 0:10]
        Y = np.expand_dims(batch[:,-1], -1)
        mask = np.expand_dims(batch[:, 10] == self.A, -1) # indicator A2 == A
        behaviour_p = self.behaviour_policy_1.predict_proba(X[:, [0,1 ,2 ,3 ,4 ,5 ,6, 8, 9]]) # compute behaviour policy
        behaviour_p = np.expand_dims(np.choose(batch[:, 10].astype(int),
                      behaviour_p.T), axis=-1)

        return {self.reward: Y, self.state: X,
                self.p_b_t1: behaviour_p, self.mask: mask}

    def train(self, use_tqdm=False, desc='None'):
        """train neural net
        Parameters
        ----------
        use_tqdm : bool
            if use tqdm
        """
        loss_value = np.zeros(self.num_iteration)
        for it in tqdm(range(self.num_iteration), disable=not use_tqdm, desc=desc):
            feed_dict = self.get_feed_dict()
            loss_value[it], _ = self.sess.run([self.loss, self.train_op], 
                                feed_dict=feed_dict)
        return loss_value

    def compute_kappa(self):
        """Compute kappa(X, a) for a = self.A and X = evaluation data
        Returns
        -------
        kappa : float [num_eval_data]
            kappa(X, a) for a = self.A and all evaluation covariates X
        """
        input_dict = self.get_eval_feed_dict()
        kappa = self.sess.run([self.out], feed_dict=input_dict)[0]
        return kappa

    def get_behaviour_policy_t0(self, X, a):
        """Predict behaviour policy probabilities for time step 0
        Parameters
        ----------
        X : np.array [None, 6]
            covariates for redicting A1
        a : int [None]
            action to compute probability of
        Returns
        -------
        probs : float [None]
            behaviour policy probability : pi_b_t0(X|a)
        """
        assert X.shape[1] == 6
        probs = self.behaviour_policy_0.predict_proba(X)
        probs = np.expand_dims(np.choose(a.astype(int), probs.T), axis=-1)
        return probs

    def get_behaviour_policy_t1(self, X, a):
        """Predict behaviour policy probabilities for time step 1
        Parameters
        ----------
        X : np.array [None, 10]
            covariates for redicting A1
        a : int [None]
            action to compute probability of
        Returns
        -------
        probs : float [None]
            behaviour policy probability : pi_b_t1(X|a)
        """
        assert X.shape[1] == 9
        probs = self.behaviour_policy_1.predict_proba(X)
        probs = np.expand_dims(np.choose(a.astype(int), probs.T), axis=-1)
        return probs
    
    def compute_lowerbound(self):
        """Compute the lower bound,
        Returns
        -------
        lower_bound : float
            computed lower bound for self.evaluation_policy
        """
        A1, A2 = self.evaluation_policy
        # setting the evaluation policy (deterministic)
        pi_e_t0 = np.zeros(2); pi_e_t0[A1] = 1
        pi_e_t1 = np.zeros(3); pi_e_t1[A2] = 1

        lower_bound =np.zeros((self.eval_data.shape[0], 1))

        for a in self.action_set:
            # behaviour policy for action a (second time step)
            pi_b_t1 = self.get_behaviour_policy_t1(self.eval_data[:, [0,1 ,2 ,3 ,4 ,5 ,6, 8, 9]], 
                                            np.ones(self.eval_data.shape[0]) * a)
            lower_bound += pi_e_t1[a] * ((1-pi_b_t1) * np.expand_dims(self.kappa[:, a], -1) +\
                     np.expand_dims((self.eval_data[:, -2]==a) * self.eval_data[:, -1], -1))
            
        pi_b_t0 = self.get_behaviour_policy_t0(self.eval_data[:, 0:6], 
                                               self.eval_data[:, 8])
        lower_bound *= np.expand_dims(pi_e_t0[self.eval_data[:, 8].astype(int)], -1)/pi_b_t0
        return np.mean(lower_bound) * self.scale_reward      

    def compute_trivial_lowerbound(self):
        """Compute the trivial lower bound,
        Returns
        -------
        lower_bound : float
            computed trivial lower bound for self.evaluation_policy
        """
        # evaluation policy
        A1, A2 = self.evaluation_policy
        # behaviour policy
        batch = self.eval_data[:, :]
        
        pi_b_t1 = self.get_behaviour_policy_t1(batch[:, [0,1 ,2 ,3 ,4 ,5 ,6, 8, 9]], 
                                            np.ones(batch.shape[0]) * A2)
        pi_b_t0 = self.get_behaviour_policy_t0(batch[:, 0:6], 
                                            np.ones(batch.shape[0]) * A1)
        # mask, A1=a1, A2=a2
        mask = np.expand_dims(np.logical_and(batch[:, 8]==A1, batch[:, 10]==A2) * 1.0, axis=-1)
        # return
        reward = np.expand_dims(batch[:, -1], axis=-1)
        # importance weight adjusted for lower bound
        weights = (1.0/pi_b_t0) * (1.0/(self.Gamma *pi_b_t1) + 1 - 1.0/self.Gamma)
        
        return np.mean(mask * reward * weights) * self.scale_reward
