import random
import numpy as np
import matplotlib.pyplot as plt

def plot_testing_steps_two_results(tester_dqprm, tester_strategy, label1='DQPRM_NearyRM', label2='DQPRM_strategyRM'):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25_dqprm = list()
    prc_50_dqprm = list()
    prc_75_dqprm = list()
    prc_25_strategy = list()
    prc_50_strategy = list()
    prc_75_strategy = list()
    steps_dqprm = list()
    steps_strategy = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    plot_dict_dqprm = tester_dqprm.results['testing_steps']
    plot_dict_strategy = tester_strategy.results['testing_steps']

    for step in plot_dict_dqprm.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))

        prc_25_dqprm.append(sum(current_25)/len(current_25))
        prc_50_dqprm.append(sum(current_50)/len(current_50))
        prc_75_dqprm.append(sum(current_75)/len(current_75))
        steps_dqprm.append(step)

    # Save the results into a csv file, where the first column is the number of steps, the second contains prc_25, the third contains prc_50, and the fourth contains prc_75
    # The first row of the .csv should contain the labels for each column
    np.savetxt('dqprm_rendezvous_results.csv', np.array([steps_dqprm, prc_25_dqprm, prc_50_dqprm, prc_75_dqprm]).T, delimiter=',', fmt='%s')

    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    for step in plot_dict_strategy.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))

        prc_25_strategy.append(sum(current_25)/len(current_25))
        prc_50_strategy.append(sum(current_50)/len(current_50))
        prc_75_strategy.append(sum(current_75)/len(current_75))
        steps_strategy.append(step)

    np.savetxt('strategy_rendezvous_results.csv', np.array([steps_strategy, prc_25_strategy, prc_50_strategy, prc_75_strategy]).T,
               delimiter=',', fmt='%s')

    plt.plot(steps_dqprm, prc_25_dqprm, alpha=0)
    plt.plot(steps_dqprm, prc_50_dqprm, color='red', label=label1)
    plt.plot(steps_dqprm, prc_75_dqprm, alpha=0)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_25_dqprm, color='red', alpha=0.25)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_75_dqprm, color='red', alpha=0.25)

    plt.plot(steps_strategy, prc_25_strategy, alpha=0)
    plt.plot(steps_strategy, prc_50_strategy, color='green', label=label2)
    plt.plot(steps_strategy, prc_75_strategy, alpha=0)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_25_strategy, color='green', alpha=0.25)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_75_strategy, color='green', alpha=0.25)

    plt.grid()
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.legend()
    plt.xscale('log')
    #plt.locator_params(axis='x', nbins=5)

    plt.savefig('rendezvous_plot.png')
    plt.show()


if __name__ == "__main__":
    # seed for cooperative buttons: 10293
    # seed for rendezvous: 18378
    np.random.seed()
    random.seed()

    num_times = 10 # Number of separate trials to run the algorithm for
    #experiment = 'buttons_diff_rm'
    experiment = 'rendezvous_diff_rm'

    if experiment == 'buttons_diff_rm':
        num_agents = 3

        from buttons_config import buttons_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = buttons_config(num_times, num_agents)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        tester_strategy = buttons_config(num_times, num_agents, strategy_rm=True)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True)

        plot_testing_steps_two_results(tester_dqprm=tester_dqprm, tester_strategy=tester_strategy)

    elif experiment == 'rendezvous_diff_rm':
        num_agents = 10

        from rendezvous_config import rendezvous_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = rendezvous_config(num_times, num_agents)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        tester_strategy = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True)

        plot_testing_steps_two_results(tester_dqprm=tester_dqprm, tester_strategy=tester_strategy)