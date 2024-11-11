import math

import matplotlib.pyplot as plt

def plot_data(file1, file2):
    # plot two txt files where data in each row is split by \t: the first column is the number of timesteps (divided by 10000), the second column the 25th percentile, the third the median, and the fourth the 75th percentile
    f = open(file1, 'r').readlines()
    steps_dict = []
    percentile_25_1, medians_1, percentile_75_1 = [], [], []
    for i in range(len(f)):
        f[i] = f[i].split('\t')
        steps_dict.append(float(f[i][0]) * 10000)

        percentile_25_1.append(float(f[i][1]))
        medians_1.append(float(f[i][2]))
        percentile_75_1.append(float(f[i][3]))

    percentile_25_1 = smooth(percentile_25_1, 0.7)
    medians_1 = smooth(medians_1, 0.7)
    percentile_75_1 = smooth(percentile_75_1, 0.7)

    f = open(file2, 'r').readlines()
    percentile_25_2, medians_2, percentile_75_2 = [], [], []
    for i in range(len(f)):
        f[i] = f[i].split('\t')

        percentile_25_2.append(float(f[i][1]))
        medians_2.append(float(f[i][2]))
        percentile_75_2.append(float(f[i][3]))

    percentile_25_2 = smooth(percentile_25_2, 0.7)
    medians_2 = smooth(medians_2, 0.7)
    percentile_75_2 = smooth(percentile_75_2, 0.7)

    plt.plot(steps_dict, medians_1, label='QL_strategyRM', color='green')
    plt.fill_between(steps_dict, percentile_25_1, percentile_75_1, alpha=0.25, color='green')

    plt.plot(steps_dict, medians_2, label='QL_ToroIcarteRM', color='red')
    plt.fill_between(steps_dict, percentile_25_2, percentile_75_2, alpha=0.25, color='red')

    plt.ylabel('Avg. Reward per Step', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.grid()
    plt.legend()
    plt.savefig('./data/ow_toro_icarte_comparison.png')


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

if __name__ == "__main__":
    plot_data('./data/summary_strategy/office-paper-experiment-Strategy.txt',
              './data/summary_toro_icarte/office-paper-experiment-Toro-Icarte.txt')