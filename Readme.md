## Off-policy Policy Evaluation Under Unobserved Confounding
This repository contains all the code necessary to replicate the results of the paper "Off-policy Policy Evaluation Under Unobserved Confounding". 

First make sure the rquirements are setup, `pip install -r requirements.txt`

### autism
Directory `autism` contains the code for Autism SMART trial experiment. `Autism.ipynb` is a notebook that generates the data for Case I and Case II of this experiment. This simulator is adopted from [Comparing Dynamic Treatment Regimes Using Repeated-Measures Outcomes: Modeling Considerations in SMART Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876020/) Appendix B.

### sepsis
The directory `sepsis` containts the code for the patinet sepsis experiments. The simulator is borrowed from [Oberst, Sontag](https://github.com/clinicalml/gumbel-max-scm). The directory contains
- `learn_policies.ipynb`: This notebook is used to generate some of the data necesary for the experiments. You can skip running this notebook by unzipping the nessecary data `unzip data/processed.zip`. The `data` directory should contain the following files:
    - `optimal_policy_st.pkl`, `mixed_policy.pkl`, `tx_tr.pkl`, `t0_policy.pkl`, `value_function.pkl`
- `sepsis_experiments.ipynb`: This notebook runs the implementation of

    1. Data genration process : That uses our confounded MDP to generate data
    2. Weighted Importance Sampling esitmates
    3. Our method and Naive lowerbound