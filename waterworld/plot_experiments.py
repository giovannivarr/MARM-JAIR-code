import math

import matplotlib.pyplot as plt
import numpy as np

def plot_compare_10_evals(file1, file2, complex_task=False):
    # plot two files containing 10 evaluations each

    max_timesteps = 10_000 if complex_task else 4_000

    steps_dict1 = {0: np.array([max_timesteps] * 10, dtype=np.float32)}
    f = open(file1, 'r').readlines()

    for i in range(len(f)):
        f[i] = f[i].split('\t')
        if len(f[i]) < 3:
            continue

        step = (int(f[i][0]) + 1) * 10_000
        if step not in steps_dict1:
            steps_dict1[step] = np.array([], dtype=np.float32)
        steps_dict1[step] = np.append(steps_dict1[step], float(f[i][2]))

    steps_dict2 = {0: np.array([max_timesteps] * 10, dtype=np.float32)}
    f = open(file2, 'r').readlines()

    for i in range(len(f)):
        f[i] = f[i].split('\t')
        if len(f[i]) < 3:
            continue

        step = (int(f[i][0]) + 1) * 10_000
        if step not in steps_dict2:
            steps_dict2[step] = np.array([], dtype=np.float32)
        steps_dict2[step] = np.append(steps_dict2[step], float(f[i][2]))

    medians_1 = np.array([np.percentile(steps_dict1[i], 50) for i in steps_dict1.keys()])
    percentile_25_1 = np.array([np.percentile(steps_dict1[i], 25) for i in steps_dict1.keys()])
    percentile_75_1 = np.array([np.percentile(steps_dict1[i], 75) for i in steps_dict1.keys()])
    medians_1 = smooth(medians_1, 0.97)
    percentile_25_1 = smooth(percentile_25_1, 0.97)
    percentile_75_1 = smooth(percentile_75_1, 0.97)

    plt.plot(list(steps_dict1.keys()), medians_1, label='PPO_strategyRM_decentralised', color='green')
    plt.fill_between(list(steps_dict1.keys()), percentile_25_1, percentile_75_1, alpha=0.25, color='green')

    medians_2 = np.array([np.percentile(steps_dict2[i], 50) for i in steps_dict2.keys()])
    percentile_25_2 = np.array([np.percentile(steps_dict2[i], 25) for i in steps_dict2.keys()])
    percentile_75_2 = np.array([np.percentile(steps_dict2[i], 75) for i in steps_dict2.keys()])
    medians_2 = smooth(medians_2, 0.97)
    percentile_25_2 = smooth(percentile_25_2, 0.97)
    percentile_75_2 = smooth(percentile_75_2, 0.97)

    plt.plot(list(steps_dict2.keys()), medians_2, label='PPO_without_strategyRM_centralised', color='red')
    plt.fill_between(list(steps_dict2.keys()), percentile_25_2, percentile_75_2, alpha=0.25, color='red')

    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.grid()
    plt.legend()
    #plt.show()
    task = "complex" if complex_task else "simple"
    plt.savefig(f'./paper_results/waterworld_{task}_task_comparison.png')

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to Tensorflow
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

if __name__ == '__main__':

    # To plot the experiments on the "complex" task
    plot_compare_10_evals('./paper_results/complex_task_decentralised.txt', './paper_results/complex_task_without_RM_centralised.txt')
    # To plot the experiments on the "simple" task
    # plot_compare_10_evals('./paper_results/simple_task_decentralised.txt', './paper_results/simple_task_without_RM_centralised.txt')