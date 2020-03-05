"""
Extra classes and functions for sepsis sim 
Most of the code are from https://github.com/clinicalml/gumbel-max-scm
"""
import numpy as np
import utils.mdptoolboxSrc.mdp as mdptools
import matplotlib.pyplot as plt

class MatrixMDP(object):

    def __init__(self, tx_mat, r_mat, p_initial_state=None, p_mixture=None):
        """__init__
        Parameters
        ----------
        tx_mat : (n_actions x n_states x n_states)  
            Transition matrix of shape 
        r_mat :  (n_actions x n_states x n_states) 
            Reward matrix of shape
        p_initial_state : Probability over initial states
        p_mixture : Probability over "mixture" components, in this case
            diabetes status
        """
        # QA the size of the inputs
        assert tx_mat.ndim == 3, \
            "Transition matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.ndim == 3, \
            "Reward matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.shape == tx_mat.shape, \
            "Transition / Reward matricies not the same shape!"
        assert tx_mat.shape[-1] == tx_mat.shape[-2], \
            "Last two dims of Tx matrix should be equal to num of states"

        # Get the number of actions and states
        n_actions = tx_mat.shape[-3]
        n_states = tx_mat.shape[-2]

        # Get the number of components in the mixture:
        # If no hidden component, add a dummy so the rest of the interface works
        if tx_mat.ndim == 3:
            n_components = 1
            tx_mat = tx_mat[np.newaxis, ...]
            r_mat = r_mat[np.newaxis, ...]
        else:
            n_components = tx_mat.shape[0]

        # Get the prior over initial states
        if p_initial_state is not None:
            if p_initial_state.ndim == 1:
                p_initial_state = p_initial_state[np.newaxis, :]

            assert p_initial_state.shape == (n_components, n_states), \
                ("Prior over initial state is wrong shape "
                 "{} != (C x S)").format(p_initial_state.shape)

        # Get the prior over components
        if n_components == 1:
            p_mixture = np.array([1.0])
        elif p_mixture is not None:
            assert p_mixture.shape == (n_components, ), \
                ("Prior over components is wrong shape "
                 "{} != (C)").format(p_mixture.shape)

        self.n_components = n_components
        self.n_actions = n_actions
        self.n_states = n_states
        self.tx_mat = tx_mat
        self.r_mat = r_mat
        self.p_initial_state = p_initial_state
        self.p_mixture = p_mixture

        self.current_state = None
        self.component = None

    def reset(self):
        """reset

        Reset the environment, and return the initial position
        Returns
        -------
        (initial state, component) : int, int
            reutrns new state and component
        """
        # Draw from the mixture
        if self.p_mixture is None:
            self.component = np.random.randint(self.n_components)
        else:
            self.component = np.random.choice(
                self.n_components, size=1, p=self.p_mixture.tolist())[0]

        # Draw an initial state
        if self.p_initial_state is None:
            self.current_state = np.random.randint(self.n_states)
        else:
            self.current_state = np.random.choice(
                self.n_states, size=1,
                p=self.p_initial_state[self.component, :].squeeze().tolist())[0]

        return self.current_state, self.component

    def step(self, action):
        """step
        Take a step with the given action

        Parameters
        ----------
        action : int
            index of the action
        
        Returns
        -------
        (next_state, reward) : (int, float)
            tupe of next state index and reward
        """
        assert action in range(self.n_actions), "Invalid action!"
        is_term = False

        next_prob = self.tx_mat[
                self.component, action, self.current_state,
                :].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"

        next_state = np.random.choice(self.n_states, size=1, p=next_prob)[0]

        reward = self.r_mat[self.component, action,
                            self.current_state, next_state]
        self.current_state = next_state

        # In this MDP, rewards are only received at the terminal state
        if reward != 0:
            is_term = True

        return self.current_state, reward, is_term

    def policyIteration(self, discount=0.9, obs_pol=None, skip_check=False,
            eval_type=1):
        """Calculate the optimal policy for the marginal tx_mat and r_mat,
        using policy iteration from pymdptoolbox

        Note that this function marginalizes over any mixture components if
        they exist.

        Parameters
        ----------
        discount : float
            Discount factor for returns
        Returns
        -------
        pi : np.array, float [n_states, n_action]
            Determninistic optimal policy 
        """
        # Define the marginalized transition and reward matrix
        r_mat_obs = self.r_mat.T.dot(self.p_mixture).T
        tx_mat_obs = self.tx_mat.T.dot(self.p_mixture).T

        # Run Policy Iteration
        pi = mdptools.PolicyIteration(
            tx_mat_obs, r_mat_obs, discount=discount, skip_check=skip_check,
            policy0=obs_pol, eval_type=eval_type)
        pi.setSilent()
        pi.run()

        # Convert this (deterministic) policy pi into a matrix format 
        pol_opt = np.zeros((self.n_states, self.n_actions))
        pol_opt[np.arange(len(pi.policy)), pi.policy] = 1

        return pol_opt

    def valueIteration(self, discount=0.99, epsilon=0.01, skip_check=False,
            max_iter=1000):
        """Calculate the value function of the optimal policy 
        for the marginal tx_mat and r_mat, using value iteration from pymdptoolbox

        Note that this function marginalizes over any mixture components if
        they exist.

        Parameters
        ----------
        discount : float
            Discount factor for returns
        Returns
        -------
        pi.V : [float]
            Value for each state
        """
        # Define the marginalized transition and reward matrix
        r_mat_obs = self.r_mat.T.dot(self.p_mixture).T
        tx_mat_obs = self.tx_mat.T.dot(self.p_mixture).T

        # Run Policy Iteration
        pi = mdptools.ValueIteration(
            tx_mat_obs, r_mat_obs, discount=discount, skip_check=skip_check,
            max_iter=max_iter, epsilon=epsilon)
        pi.setSilent()
        pi.run()

        return pi.V

def plot_design_sensitivity(results, linewidth=3, color='k', fontsize=25, title='', 
                            yticks=None, xlim=None, ylim=None, alpha=0.2, scale=0.8):
    """ plotting function for design sensitivty experiment
    
    Parameters
    ----------
    results : dictionary with key, value pairs
        GAMMAs : np.array [float]
            confounding levels
        pol1_lower, pol2_lower : np.array [float]
            lower bound on policy 1, 2
        pol1_upper, pol2_upper : np.array [float]
            upper bound on policy 1, 2
        pol1_label, pol2_label : string
            label on policy 1,2 
        cross : float
            crossing value of two policies
    """
    # Policy 1
    plt.plot(results['GAMMAs'], results['pol1_lower'],color=color, linewidth=linewidth)
    plt.plot(results['GAMMAs'], results['pol1_upper'],color=color, linewidth=linewidth)
    plt.fill_between(results['GAMMAs'], results['pol1_upper'], results['pol1_lower'], 
                            alpha=alpha, color=color)
    value = 0.5 * (results['pol1_lower'][0] + results['pol1_upper'][0])
    plt.plot(results['GAMMAs'], [value]*len(results['GAMMAs']), linestyle='solid', color=color,
                            alpha=1-alpha, linewidth=linewidth*scale, label=results['pol1_label'])
    # Policy 2
    plt.plot(results['GAMMAs'], results['pol2_lower'],color=color, linewidth=linewidth)
    plt.plot(results['GAMMAs'], results['pol2_upper'],color=color, linewidth=linewidth)
    plt.fill_between(results['GAMMAs'], results['pol2_upper'], results['pol2_lower'], 
                            alpha=alpha, color=color)
    value = 0.5 * (results['pol2_lower'][0] + results['pol2_upper'][0])
    plt.plot(results['GAMMAs'], [value]*len(results['GAMMAs']), linestyle='dashed', color=color,
                            alpha=1-alpha, linewidth=linewidth*scale, label=results['pol2_label'])
    

    plt.plot([results['cross']]*2, ylim, color=color, linestyle='dashed', linewidth=linewidth/scale)

    #plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.xlabel(r'Level of confounding ($\Gamma$)', fontsize=fontsize)
    plt.ylabel(r'Outcome $\mathbb{E}[Y(\bar{A}_{1:T})]$', fontsize=fontsize)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(fontsize=fontsize*scale, loc='best')
    plt.title(title, fontsize=fontsize, y=1.05)