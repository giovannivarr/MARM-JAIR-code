# Synthesising Reward Machines for Cooperative Multi-Agent Reinforcement Learning 
## Neary et al. & Toro-Icarte et al.
This folder contains the code and data related to the comparisons with previous work in the literature we present in the paper.

### Neary et al. 
The `neary_et_al` folder contains the code necessary to run the comparison with the approach for multi-agent reinforcement learning with reward machines presented in "Reward Machines for Cooperative Multi-Agent Reinforcement Learning" by Neary et al. Most of the code comes from the original repository of the paper, which can be found at [`github.com/cyrusneary/rm-cooperative-marl`](https://github.com/cyrusneary/rm-cooperative-marl). 

To run an experiment (as presented in the paper), it suffices to run the Python script `run_compare.py` which can be found in the `src` folder. In the script, it is possible to choose between the CooperativeButtons and the Rendezvous by setting the variable `experiment` in the script. After training the two sets of agents, the script will also plot their results (in the same style as the plot in the paper) and save the results in two `.csv` files: `dqprm_results.csv` and `strategy_results.csv`. It is also possible to save the plot as `.png` file, by uncommenting line 101 of the script.   

We provide both seeds and the data that were used to obtain the plots in the paper. The seeds can be found at lines 105 and 106 (respectively, for the CooperativeButtons and Rendezvous experiments), whereads the data is in the `data` folder. Files ending with `buttons_results` and with `rendezvous_results` contain the data of the respective experiments, whereas files starting with `dqprm` contain the data obtained by using the reward machines of Neary et al., and the files starting with `strategy` contain the data obtained by using the reward machines we have synthesised.


### Toro Icarte et al.

The `toro_icarte_et_al` folder contains the code necessary to run the comparison with the approach presented in original paper about reward machines, "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning" by Toro Icarte et al. As for the Neary et al. code, most of the code in this folder comes from the original repository of the paper, which can be found at [`github.com/RodrigoToroIcarte/reward_machines`](https://github.com/RodrigoToroIcarte/reward_machines).

To run an experiment (as presented in the paper), it suffices to launch the `run_office_experiment.sh` script, which can be found in the main folder. By default, the script saves the data of the two runs in the folder `data/ql/office-ENV/experiment/` (where `ENV` is `Toro-Icarte` and `Strategy` for the respective results). The destination folder can be changed by modifying the `--log_path` option in the script. 

It is possible to process the experiment data by running the `process_results.py` script in the `src` folder. By default, this will process the data stored in `paper-experiment` folders (i.e., the ones used to obtain the paper's plots). After obtaining the processed data, it can be used to produce plots (e.g., by using Matplotlib); this can be done by using the `plot_eval.py` script, and by specifying the paths to the processed data files to use to draw the plot. 

Finally, in the `data/ql/office-ENV/paper-experiment` folders (where `ENV` is `Toro-Icarte` and `Strategy` for the respective experiments), we provide the data that was used to obtain the plots in the paper.



### Dependencies and Python versions
The experiments in the `neary_et_al` folder can be run with Python 3.11 and Python 3.12. On the other hand, the experiments in the `toro_icarte_et_al` folder can be run only with Python 3.6 or 3.7, due to the required libraries. Each folder has its own `requirements.txt` file: we advise to create a dedicated virtual environment and install all the necessary dependancies via such files. 