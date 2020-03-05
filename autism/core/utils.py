"""Plotting and other functions for autsim experiment"""

import numpy as np
import matplotlib.pyplot as plt

def seprate_policies(data, slow_responder=True):
    """sperates the data for two different policies
    1. AAC, 2. BLI + AAC
    Parameters
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients
    Returns
    -------
    pi_aac : np.array [None, num_covariates + num_outputs]
        part of data that are assigned to policy AAC
    pi_adaptive : np.array [None, num_covariates + num_outputs]
        part of data that are assigned to policy AAC + BLI
    pi_int : np.array [None, num_covariates + num_outputs]
        part of data that are assigned to policy BLI and Intesified BLI
    """
    if slow_responder:
        data = data[data[:, 7] == 0, :]
    pi_aac = data[data[:, 8] == -1, :]

    pi_adaptive = data[data[:, 8] == 1, :]
    pi_adaptive = pi_adaptive[pi_adaptive[:, 9] == -1, :]

    pi_int = data[data[:, 8] == 1, :]
    pi_int = pi_int[pi_int[:, 9] == 1, :]

    return pi_aac, pi_int, pi_adaptive

def compute_effect_size(data, slow_responder=True):
    """computes the effect size of the adaptive policy
    as descibed in appendix B
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876020/

    Parameters 
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients

    Returns
    -------
    effect_size : float
        effect size observed in :param data:
    """
    _, pi_int, pi_adaptive = seprate_policies(data, slow_responder=slow_responder)

    AUC11 = (pi_int[:, 10]/2 + pi_int[:, 11] + pi_int[:, 12] + pi_int[:,13]/2) * 12
    AUC1n1 = (pi_adaptive[:, 10]/2 + pi_adaptive[:, 11] + pi_adaptive[:, 12] + pi_adaptive[:,13]/2) * 12
    
    m11 = np.mean(AUC11); s11=np.std(AUC11)
    m1n1 = np.mean(AUC1n1); s1n1=np.std(AUC1n1)
    return (m1n1 - m11)/np.sqrt((s11**2 + s1n1**2)/2)

def compute_observational_policy_value(data, slow_responder=True):
    """computes observational estimate of the policy value
    Parameters 
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    slow_responder : bool
        if True, only looks at slow responder patients

    Returns
    -------
    pi_acc_v : float
        value of AAC policy
    pi_adaptive_v : float
        value of AAC + BLI policy
    """ 
    pi_aac, _, pi_adaptive = seprate_policies(data, slow_responder=slow_responder)
    pi_acc_v = np.concatenate([[np.mean(data[:, 10])], 
                                np.mean(pi_aac[:,11:14], axis=0)])
    pi_adaptive_v = np.concatenate([[np.mean(data[:, 10])], 
                                     np.mean(pi_adaptive[:,11:14], axis=0)])
    
    return pi_acc_v, pi_adaptive_v

def plot_autism(data, evaluations=None, title='', fontsize=30, 
                      markersize=18, linewidth=6, fontscale=0.6):
    """genrates plot for autism example
    Parameters
    ----------
    data : float np.array num_data x (num_covaraites + num_outputs)
        this is the output of DataGen class -- see data_gen.py
    evaluations : dictionary
        key : legend of the observation, eg. 'BLI+AAC'
        value : (int, float, color, marker)
            tuple of week, value, point color, and marker type
    confounding : float
        amount of confounding in data generation process
    """
    
    weeks = [0, 12, 24, 36]
    # Plot observational value
    pi_aac, pi_adaptive = compute_observational_policy_value(data, slow_responder=True)
    plt.plot(weeks, pi_aac, '-o', color = 'black', 
                markersize=markersize, label='AAC', linewidth=linewidth)
    plt.plot(weeks, pi_adaptive, '--p', color = 'black', 
                markersize=markersize, label='BLI + AAC (OPE)', linewidth=linewidth)
    # Plot evaluated values
    if evaluations is not None:
        for key in evaluations.keys():
            week, value, color, marker = evaluations[key]
            plt.plot(week, value, marker, color=color, 
                    markersize=markersize, label=key, linewidth=linewidth, 
                    fillstyle='full')

    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize*fontscale)
    plt.xlabel("Weeks", fontsize=fontsize)
    plt.ylabel(r"$\mathbb{E}[Y(\bar A_{1:T})]$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize*fontscale)
    plt.yticks(fontsize=fontsize*fontscale)
    plt.grid(linewidth=3)

def plot_autism_design_sensitivity(data, title='', fontsize=30, 
                      markersize=23, linewidth=6, fontscale=0.8):
    """genrates plot for autism example, design sensitivity 
    Parameters
    ----------
    data : dictionary with four (key, value)
        - 'our' : float, np.array : computed lower bounds with our method
        - 'naive' : float, np.array : computed lower bounds with naive bounds
        - 'Gammas' : float, np.array
        - 'aac' : float : value of aac policy
        - 'adaptive' : float : value of adaptive policy
    """
    length = len(data['Gammas'])
    aac = np.array([data['aac']] * length)
    adaptive = np.array([data['adaptive']] * length)

    plt.plot(data['Gammas'], aac, '-o', color = 'black', 
                markersize=markersize, label='AAC (True)', linewidth=linewidth)
    plt.plot(data['Gammas'], adaptive, '-p', color = 'green', 
                markersize=markersize, label='BLI + AAC (True)', linewidth=linewidth)
    plt.plot(data['Gammas'], data['our'], '-.p', color = 'blue', 
                markersize=markersize, label='BLI + AAC (Ours)', linewidth=linewidth)
    plt.plot(data['Gammas'], data['naive'], '-.p', color = 'red', 
                markersize=markersize, label='BLI + AAC (Naive)', linewidth=linewidth)
    
    ymax = data['adaptive'] + 2.0
    ymin = np.min(data['our']) - 10.0

    plt.vlines([data['cross'][0]] , ymax, ymin, 
            linestyles='--', colors='gray', linewidth=linewidth)
    plt.vlines([data['cross'][1]] , ymax, ymin,  
            linestyles='--', colors='gray', linewidth=linewidth)

    plt.title(title, 
                fontsize=fontsize, y=1.02)
    plt.legend(loc=4, fontsize=fontsize*fontscale)
    plt.xlabel("Level of confounding ($\Gamma$)", fontsize=fontsize)
    plt.ylabel(r"outcome $\mathbb{E}[Y(\bar A_{1:T})]$", fontsize=fontsize)
    plt.xticks(data['Gammas'].tolist() + [data['cross'][0], data['cross'][1]], 
            fontsize=fontsize*fontscale)
    plt.yticks(fontsize=fontsize*fontscale)
    plt.ylim([ymin, ymax])
    plt.grid(linewidth=3)