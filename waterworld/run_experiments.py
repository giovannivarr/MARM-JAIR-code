import os
import argparse
import torch
import random
import numpy as np
import ray

from copy import copy
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN
from ray.tune import CLIReporter
from reward_machines.reward_machine import RewardMachine
from ray.rllib.algorithms import Algorithm, PPOConfig
from ray.rllib.policy.policy import Policy
from marm_waterworld.single_agent import sarm_waterworld
from marm_waterworld.multi_agent import marm_waterworld
from ray import tune, train
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.train import CheckpointConfig

os.environ["RAY_DEDUP_LOGS"] = "0"

parser = add_rllib_example_script_args(
    default_iters=10_000_000,
    default_timesteps=1_000_000
)

#rm_path_0 = './marm_waterworld/reward_machines/rm_simple_0.txt'
#rm_path_1 = './marm_waterworld/reward_machines/rm_simple_1.txt'
rm_path_0 = './marm_waterworld/reward_machines/rm_complex_0.txt'
rm_path_1 = './marm_waterworld/reward_machines/rm_complex_1.txt'

reward_machines = [RewardMachine(rm_path_0), RewardMachine(rm_path_1)]


def individual_training_and_evaluation():
    config_agent0 = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment("SingleAgent-0-WaterWorld")
        .env_runners(num_env_runners=4, num_cpus_per_env_runner=1, sample_timeout_s=None)
        .learners(num_learners=1, num_cpus_per_learner=1)
        .training(
            vf_loss_coeff=0.7,
            clip_param=0.1,
            lr=[[0, 1e-5], [3_000_000, 1e-9]],
            gamma=0.99,
            lambda_=0.99,
            num_sgd_iter=20,
            mini_batch_size_per_learner=250,
            train_batch_size=10_000,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512],
                #"use_lstm": True
            },
        )
    )

    config_agent1 = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment("SingleAgent-1-WaterWorld")
        .env_runners(num_env_runners=4, num_cpus_per_env_runner=1, sample_timeout_s=None)
        .learners(num_learners=1, num_cpus_per_learner=1)
        .training(
            vf_loss_coeff=0.7,
            clip_param=0.1,
            lr=[[0, 1e-5], [3_000_000, 1e-9]],
            gamma=0.99,
            lambda_=0.99,
            num_sgd_iter=20,
            mini_batch_size_per_learner=250,
            train_batch_size=10_000,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512],
                #"use_lstm": True
            },
        )
    )

    checkpointer = CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=1)

    results = tune.run("PPO", config=config_agent0, stop={"timesteps_total": 10_000_000},
                       checkpoint_config=checkpointer)
    checkpoint = results.get_last_checkpoint()
    policy_ag0 = Policy.from_checkpoint(checkpoint.path)

    results = tune.run("PPO", config=config_agent1, stop={"timesteps_total": 10_000_000},
                       checkpoint_config=checkpointer)
    checkpoint = results.get_last_checkpoint()
    policy_ag1 = Policy.from_checkpoint(checkpoint.path)

    policy_agents = {'pursuer_0': list(policy_ag0.items())[0][1].get_weights(),
                     'pursuer_1': list(policy_ag1.items())[0][1].get_weights()}

    test_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .env_runners(num_env_runners=4, num_cpus_per_env_runner=7, sample_timeout_s=None)
        .environment("WaterWorld")
        .multi_agent(
            policies={"pursuer_0", "pursuer_1"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512]
            },
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "pursuer_0": SingleAgentRLModuleSpec(),
                    "pursuer_1": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    # now load the policies in policy_agents into the config
    test = test_config.build()

    # load policies into test
    test.set_weights(policy_agents)
    print(test.evaluate())


def centralised_training_and_evaluation():
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .env_runners(num_env_runners=4, num_cpus_per_env_runner=7, sample_timeout_s=None)
        .learners(num_learners=3, num_cpus_per_learner=1)
        .environment("WaterWorld")
        .multi_agent(
            policies={"pursuer_0", "pursuer_1"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .training(
            vf_loss_coeff=0.7,
            clip_param=0.1,
            lr=[[0, 1e-5], [6_000_000, 1e-9]],
            gamma=0.99,
            #use_gae=False,
            lambda_=0.99,
            num_sgd_iter=20,
            mini_batch_size_per_learner=500,
            train_batch_size=20_000,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512]
            },
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "pursuer_0": SingleAgentRLModuleSpec(),
                    "pursuer_1": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    checkpointer = CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=1)

    results = tune.run("PPO", config=config, stop={"timesteps_total": 20_000_000},
                       checkpoint_config=checkpointer)
    checkpoint = results.get_last_checkpoint()

    test = Algorithm.from_checkpoint(checkpoint.path)
    print(test.evaluate())


def tuning():
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["mini_batch_size_per_learner"] * 2:
            config["train_batch_size"] = config["mini_batch_size_per_learner"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(10, 30),
        "mini_batch_size_per_learner": lambda: random.randint(1000, 8000),
        "train_batch_size": lambda: random.randint(2000, 20000),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when either we've reached mean reward >= 0.7 with at least 20_000 timesteps elapsed, or when at least 50 million timesteps have elapsed 
    def stop_fn(trial_id: str, result: dict) -> bool:
        return result["training_iteration"] == 50
        # return (result["env_runners"]["episode_return_mean"] >= 0.7 and result["timesteps_total"] >= 20_000) or result["timesteps_total"] >= 50_000_000 

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_reward_mean",
            mode="max",
            # scheduler=pbt,
            num_samples=1,
        ),
        param_space={
            "env": "SingleAgent-0-WaterWorld",
            "num_workers": 7,  # number of CPUs to use for collecting samples
            "num_cpus": 1,  # number of CPUs to use per trial,
            "vf_loss_coeff": 0.005,
            # These params are tuned from a fixed starting value.
            "lambda": 0.99,
            # Grid search for params
            "lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
            "clip_param": tune.grid_search([0.1, 0.2, 0.3]),
            "model": {"fcnet_hiddens": [512, 512, 512]},
            "num_sgd_iter": tune.grid_search([10, 20, 30]),
            "mini_batch_size_per_learner": tune.grid_search([250, 500, 1000]),
            "train_batch_size": tune.grid_search([5_000, 10_000, 15_000]),
        },
        run_config=train.RunConfig(stop=stop_fn),
    )
    results = tuner.fit()

    import pprint

    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = {
        "episode_reward_mean": best_result.metrics["env_runners"]["episode_reward_mean"],
        "episode_reward_max": best_result.metrics["env_runners"]["episode_reward_max"],
        "episode_reward_min": best_result.metrics["env_runners"]["episode_reward_min"],
        "episode_len_mean": best_result.metrics["env_runners"]["episode_len_mean"],
    }
    pprint.pprint(metrics_to_print)


def evaluate_decentralised_agents(checkpoint_path_0, checkpoint_path_1, num_checkpoints=100, num_eval_episodes=1, seed=None):
    if seed is not None:
        print('Seed: ', seed)
        register_env(
            "WaterWorldEval",
            lambda _: PettingZooEnv(
                marm_waterworld.env(reward_machines=reward_machines, reward_shaping=args.rewardshaping,
                                    speed_features=args.speedfeatures, rm_less=args.rmless,
                                    max_cycles=args.maxcycles, seed=seed))
        )
    else:
        register_env(
            "WaterWorldEval",
            lambda _: PettingZooEnv(
                marm_waterworld.env(reward_machines=reward_machines, reward_shaping=args.rewardshaping,
                                    speed_features=args.speedfeatures, rm_less=args.rmless, max_cycles=args.maxcycles))
        )
    
    test_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment("WaterWorldEval")
        .multi_agent(
            policies={"pursuer_0", "pursuer_1"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=num_eval_episodes,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512]
            },
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "pursuer_0": SingleAgentRLModuleSpec(),
                    "pursuer_1": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    test = test_config.build()

    for i in range(num_checkpoints):
        checkpoint_0 = checkpoint_path_0 + 'checkpoint_' + str(i).zfill(6)
        checkpoint_1 = checkpoint_path_1 + 'checkpoint_' + str(i).zfill(6)
        policy_ag0 = Policy.from_checkpoint(checkpoint_0)
        policy_ag1 = Policy.from_checkpoint(checkpoint_1)
        policy_agents = {'pursuer_0': list(policy_ag0.items())[0][1].get_weights(),
                         'pursuer_1': list(policy_ag1.items())[0][1].get_weights()}

        # load policies into test
        test.set_weights(policy_agents)
        res = test.evaluate()
        ## now print the episode_reward_mean and episode_len_mean
        print(i, "\t", res['env_runners']['episode_reward_mean'], "\t", res['env_runners']['episode_len_mean'], "\t", res['env_runners']['hist_stats']['episode_lengths'])


def evaluate_centralised_agents(checkpoint_path, num_checkpoints=100, num_eval_episodes=1, seed=None):
    if seed is not None:
        print(seed)
        register_env(
            "WaterWorldEval",
            lambda _: PettingZooEnv(
                marm_waterworld.env(reward_machines=reward_machines, reward_shaping=args.rewardshaping, 
                                    speed_features=args.speedfeatures, rm_less=args.rmless, 
                                    max_cycles=args.maxcycles, seed=seed))
        )
    else:
        register_env(
            "WaterWorldEval",
            lambda _: PettingZooEnv(
                marm_waterworld.env(reward_machines=reward_machines, reward_shaping=args.rewardshaping, 
                                    speed_features=args.speedfeatures, rm_less=args.rmless, max_cycles=args.maxcylces))
        )
    
    test_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment("WaterWorldEval")
        .multi_agent(
            policies={"pursuer_0", "pursuer_1"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=num_eval_episodes,
            evaluation_duration_unit="episodes",
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [512, 512, 512]
            },
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "pursuer_0": SingleAgentRLModuleSpec(),
                    "pursuer_1": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    test = test_config.build()

    for i in range(num_checkpoints):
        checkpoint = checkpoint_path + 'checkpoint_' + str(i).zfill(6)
        policy = Policy.from_checkpoint(checkpoint)
        policy_agents = {'pursuer_0': list(policy.items())[0][1].get_weights(),
                         'pursuer_1': list(policy.items())[0][1].get_weights()}

        # load policies into test
        test.set_weights(policy_agents)
        res = test.evaluate()
        # now print the episode_reward_mean and episode_len_mean
        print(i, "\t", res['env_runners']['episode_reward_mean'], "\t", res['env_runners']['episode_len_mean'], "\t", res['env_runners']['hist_stats']['episode_lengths'])


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser(description='Script to run independent single-agent training or centralised '
                                                 'multi-agent training')
    parser.add_argument('--centralised', help='train single agent or multi agent', action='store_true')
    parser.set_defaults(centralised=False)
    parser.add_argument('--tuning', help='perform hyperparameter tuning', action='store_true')
    parser.set_defaults(tuning=False)
    parser.add_argument('--evaluation', help='perform agent evaluation', action='store_true')
    parser.set_defaults(tuning=False)
    parser.add_argument('--rewardshaping', help='use reward shaping', action='store_true')
    parser.set_defaults(rewardshaping=False)
    parser.add_argument('--speedfeatures', help='use speed features in Waterworld', action='store_true')
    parser.set_defaults(speedfeatures=False)
    parser.add_argument('--rmless', help='disable access to reward machine state', action='store_true')
    parser.set_defaults(rmless=False)
    parser.add_argument('--maxcycles', help='specify maximum number of timesteps in an episode', type=int, default=2_000)

    args = parser.parse_args()

    register_env(
        "SingleAgent-0-WaterWorld",
        lambda _: PettingZooEnv(
            sarm_waterworld.env(agent_id=0, reward_machine=reward_machines[0], reward_shaping=args.rewardshaping, speed_features=args.speedfeatures, rm_less=args.rmless, max_cycles=args.maxcycles))
    )

    register_env(
        "SingleAgent-1-WaterWorld",
        lambda _: PettingZooEnv(
            sarm_waterworld.env(agent_id=1, reward_machine=reward_machines[1], reward_shaping=args.rewardshaping, speed_features=args.speedfeatures, rm_less=args.rmless, max_cycles=args.maxcycles))
    )

    register_env(
        "WaterWorld",
        lambda _: PettingZooEnv(
            marm_waterworld.env(reward_machines=reward_machines, reward_shaping=args.rewardshaping, speed_features=args.speedfeatures, rm_less=args.rmless, max_cycles=args.maxcycles))
    )

    if args.tuning:
        tuning()
    elif args.evaluation:
        seeds = [1234, 1342, 42, 0, 10, 1000, 24, 342, 243, 432]
        if args.centralised:
            for seed in seeds:
                continue
                # Agents trained with the centralised approach (either with or without RMs)
                # Input path to the folder containing the PPO policy of the trained agents 
                # evaluate_centralised_agents("", seed=seed, num_checkpoints=1_000)
        else:
            for seed in seeds:
                continue
                # Agents trained with the decentralised approach
                # Input paths to the folders containing the PPO policies of the trained agents
                # evaluate_decentralised_agents("", "", seed=seed, num_checkpoints=1_000)

    else:
        if args.centralised:
            centralised_training_and_evaluation()
        else:
            individual_training_and_evaluation()
