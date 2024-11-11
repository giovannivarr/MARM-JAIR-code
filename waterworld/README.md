# Synthesising Reward Machines for Cooperative Multi-Agent Reinforcement Learning 
## WaterWorld
This folder contains the code necessary to run the experiments in WaterWorld. We have used two libraries to develop these experiments. 

Our environment is a modified version of the WaterWorld implementation [PettingZoo](https://pettingzoo.farama.org). We have modified the environment in two ways: (i), to allow for the agents to be trained individually, and (ii), to add reward machines to the environment.

We have used the [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) library to import the implementation of PPO that we used to train the agents. We used the RayRLlib PPO implementation off-the-shelf, without any modification to the algorithm. 

### Training agents
The experiments can be run with Python 3.11 and Python 3.12. We advise to create a virtual environment and use the `requirements.txt` file in the folder to install all the necessary libraries. In order to run the experiments, open a terminal, set its working directory to be the `waterworld` folder, and then run the `run_experiments.py` script. Following is an explanation of the possible options of the script.


#### Usage
```
WaterWorld usage: run_experiments.py [--centralised] [--tuning] [--evaluation]
				     [--rewardshaping] [--speedfeatures] [--rmless]
				     [--maxcycles MAXCYCLES]

optional arguments:
	--centralised		train agents with the centralised approach 
				(by default agents are trained with the decentralised approach - in the experiments this is only used in conjunction with --rmless)
	--tuning		perform hyperparameter tuning 
				(NOTE: hyperparameter tuning is performed with the decentralised approach)
	--evaluation	        evaluate the agents' performance; if --centralised is also set, 
				then evaluate the performance of agents trained with the centralised approach
	--rewardshaping		enable reward shaping for the reward machines
	--speedfeatures		enable agents to observe the speed features of the evaders
	--rmless		disable agents to access the reward machine and its intermediate rewards
	--maxcycles MAXCYCLES	set the maximum number of timesteps per agent [default: 2000]
	
```

Note that, in order to evaluate trained agents, it is necessary to specify the path(s) to the policies saved by Ray RLlib after training. These must be specified either on line 452 (to evaluate agents trained with the centralised approach), or on line 458 (to evaluate agents trained with the decentralised approach). By default, the evaluation prints the results on the terminal; it is possible to save these into a file by running the following command in the terminal:

	run_experiments.py --evaluation [--centralised] > /path/to/results/file


### Plotting the data
Data obtained by evaluating agents can be plot by using the `plot_experiments.py` script. To plot the data, as to train agents, open a terminal and set its working directory to be the `waterwolrd` folder. Then, it suffices to run the `plot_experiments.py` script. 

The path to the data files can be specified on line 106. Note that it is only possible to plot data if three files are provided: one for agents trained with the decentralised approach, one for agents trained with the centralised approach, and one for agents trained without access to the reward machines and with the centralised approach. We already include the code necessary to obtain the plots appearing in the paper: on line 106 we plot the results for the "complex" task, and on line 108 for the simple "task". By changing the file paths on these lines, it is possible to obtain new plots with different data. 

By default, the script saves the output plot in a file named `waterworld.png`. It is possible to change the output file name by setting it on line 82. 

