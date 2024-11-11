import math, os, statistics, argparse
from collections import deque
import numpy as np


def get_precentiles_str(a):
    p25 = "%0.4f" % float(np.percentile(a, 25))
    p50 = "%0.4f" % float(np.percentile(a, 50))
    p75 = "%0.4f" % float(np.percentile(a, 75))
    return [p25, p50, p75]


def export_avg_results_grid_single(env, experiment, seeds):
    """
    Export the average results for the OfficeWorld environment.
    Parameters
    ----------
    env: string indicating whether to process the data for the "Toro-Icarte" or "Strategy" experiment
    experiment: string indicating the folder within "data/ql/env/" where the results to be exported are stored
    seeds: list of seeds to be considered

    """

    num_episodes_avg = 100
    num_total_steps = 1e5
    max_length = 100
    # These values were computed using
    # python3 test_optimal_policies.py --env Office-single-Toro-Icarte
    # python3 test_optimal_policies.py --env Office-single-Strategy
    if env == 'Toro-Icarte':
        optimal_reward = 0.03130872235829314  # toro-icarte
    elif env == 'Strategy':
        optimal_reward = 0.021114329408753647  # strategy
    else:
        assert False, "Invalid environment!"

    stats = [[] for _ in range(max_length)]

    for seed in seeds:
        # Reading the results
        if 'Toro-Icarte' in env:
            f_path = "../data/ql/office-Toro-Icarte/%s/%s/0.0.monitor.csv" % (experiment, seed)  # toro-icarte
        elif 'Strategy' in env:
            f_path = "../data/ql/office-Strategy/%s/%s/0.0.monitor.csv" % (experiment, seed)  # strategy
        results = []
        f = open(f_path)
        for l in f:
            raw = l.strip().split(',')
            if len(raw) != 3 or raw[0] == 'r':
                continue
            r, l, t = float(raw[0]), float(raw[1]), float(raw[2])
            results.append((t, l, r))
        f.close()

        # collecting average stats
        steps = 0
        rewards = deque([], maxlen=num_episodes_avg)
        steps_tic = num_total_steps / max_length
        for i in range(len(results)):
            _, l, r = results[i]
            rew_per_step = (r / l) / optimal_reward
            if (steps + l) % steps_tic == 0:
                steps += l
                rewards.append(rew_per_step)
                stats[int((steps + l) // steps_tic) - 1].append(sum(rewards) / len(rewards))
            else:
                if (steps // steps_tic) != (steps + l) // steps_tic:
                    stats[int((steps + l) // steps_tic) - 1].append(sum(rewards) / len(rewards))
                steps += l
                rewards.append(rew_per_step)
            if (steps + l) // steps_tic == max_length:
                break

    # Saving the average performance and standard deviation
    if env == 'Toro-Icarte':
        f_out = "../data/summary_toro_icarte/office-%s-Toro-Icarte.txt" % experiment # toro-icarte
    elif env == 'Strategy':
        f_out = "../data/summary_strategy/office-%s-Strategy.txt" % experiment  # strategy

    # Write the output file, containing the number of steps (divided by 1000), and the 25th, 50th, and 75th percentiles
    f = open(f_out, 'w')
    for i in range(max_length):
        if len(stats[i]) == len(seeds) * len(maps):
            f.write("\t".join([str((i + 1) * steps_tic / 1000)] + get_precentiles_str(stats[i])) + "\n")
    f.close()


if __name__ == '__main__':
    print('ql: office-Toro-Icarte')
    export_avg_results_grid_single(env='Toro-Icarte', experiment='paper-experiment', seeds=list(range(60)))
    print('ql: office-Strategy')
    export_avg_results_grid_single(env='Strategy', experiment='paper-experiment', seeds=list(range(60)))
