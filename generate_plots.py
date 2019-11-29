import matplotlib.pyplot as plt
import numpy as np
import sys

def get_val(s, s1, s2):
    start = s.find(s1)+len(s1)
    end = s.find(s2)
    reward = float(s[start:end])
    return reward

def get_smoothed_val(y, size=100):
    y_new = [y[0]]*(size-1)
    y_new = y_new + y
    y_new = y_new + [y[len(y)-1]]*(size-1)
    filter_y = np.ones(size)/size
    ans = np.convolve(y_new, filter_y,mode='same')
    return ans[(size-1):len(ans)-(size-1)]

def clip_spikes(y, min_y, max_y):
    return np.clip(y, a_min=min_y, a_max=max_y)

def get_episode_values(txtfile):
    episodes = list()
    rewards = list()
    waiting_times = list()
    count = 0
    with open(txtfile) as file:
        for line in file:
            reward = get_val(line, "TOTAL REWARD: ", ", AVGWAITTIME: ")
            waiting_time = get_val(line, ", AVGWAITTIME: ", "\n")
            rewards.append(reward)
            waiting_times.append(waiting_time)
            episodes.append(count)
            count += 1
    return episodes, rewards, waiting_times

def get_epsilon_values(txtfile):
    steps = list()
    epsilons = list()
    count = 0
    with open(txtfile) as file:
        for line in file:
            steps.append(count)
            epsilon = get_val(line, "STEP: ", "\n")
            epsilons.append(epsilon)
            count += 1
    return steps, epsilons

def get_status_values(txtfile):
    iters = list()
    td_errors = list()
    count = 0
    with open(txtfile) as file:
        for line in file:
            iters.append(count)
            td_loss = get_val(line, ": TDLOSS: ", "\n")
            td_errors.append(td_loss)
            count += 1
    return iters, td_errors

flag = 0
episode_file = "Logs/3dqn_episode.txt"
epsilon_file = "Logs/3dqn_epsilon.txt"
status_file = "Logs/3dqn_status.txt"

mapping = dict()
mapping["Logs/3dqn_episode_harsh_mail.txt"] = "Vanilla DQN"
mapping["Logs/3dqn_episode_priority_sampling.txt"] = "D3QN with Priority Sampling"
mapping["Logs/3dqn_episode_uniform_sampling.txt"] = "D3QN with Uniform Sampling"
mapping["Logs/out_fixed5_5_5_5_episode.txt"] = "Fixed time - 5 seconds"
mapping["Logs/out_fixed15_15_15_15_episode.txt"] = "Fixed time - 15 seconds"
mapping["Logs/out_fixed30_30_30_30_episode.txt"] = "Fixed time - 30 seconds"
mapping["Logs/out_fixed45_45_45_45_episode.txt"] = "Fixed time - 45 seconds"
mapping["Logs/out_fixed60_60_60_60_episode.txt"] = "Fixed time - 60 seconds"

if len(sys.argv) == 0:
    episode_file = "Logs/dqn_episode.txt"
    epsilon_file = "Logs/dqn_epsilon.txt"
    status_file = "Logs/dqn_status.txt"
    flag = 1
else:
    if sys.argv[len(sys.argv)-1] == 0:
        episode_file = sys.argv[1]
        epsilon_file = sys.argv[2]
        status_file = sys.argv[3]
        flag = 1
    else:
        rewards_arr = list()
        waiting_time_arr = list()
        episodes_arr = list()
        algos = list()
        for i in range(1, len(sys.argv)):
            episodes, rewards, waiting_times = get_episode_values(sys.argv[i])
            episodes_arr.append(episodes)
            rewards_arr.append(rewards)
            waiting_time_arr.append(waiting_times)
            algos.append(sys.argv[i])
        plt.figure()
        for j in range(len(rewards_arr)):
            rewards = rewards_arr[j]
            if algos[j][5] != 'o':
                rewards = get_smoothed_val(rewards_arr[j])
            else:
                #rewards = get_smoothed_val(rewards_arr[j], 5)
                pass
            plt.plot(episodes_arr[j], rewards, label=mapping[algos[j]])
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("Total Reward")
        plt.yticks([-350000, -325000, -300000, -275000, -250000, -225000, -200000, -175000, -150000], ['-3.5L', '-3.25L', '-3.0L', '-2.75L', '-2.5L', '-2.25L', '-2.0L', '-1.75L', '-1.5L'])
        plt.savefig("Plots/combined_reward_plot.png")
        plt.show()
        plt.close()
        plt.figure()
        for k in range(len(waiting_time_arr)):
            waiting_times = waiting_time_arr[k]
            if algos[k][5] != 'o':
                waiting_times = get_smoothed_val(waiting_time_arr[k])
            else:
                #waiting_times = get_smoothed_val(waiting_time_arr[k], 5)
                pass
            plt.plot(episodes_arr[k], waiting_times, label=mapping[algos[k]])
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Waiting Time")
        plt.savefig("Plots/combines_waiting_time_plot.png")
        plt.show()
        plt.close()

if flag == 1:
    episodes, rewards, waiting_times = get_episode_values(episode_file)
    steps, epsilons = get_epsilon_values(epsilon_file)
    iters, td_errors = get_status_values(status_file)

    # Plot Reward v/s Episodes
    rewards = get_smoothed_val(rewards)
    plt.plot(episodes, rewards)
    plt.xlabel("Number of episodes")
    plt.yticks([-320000, -300000, -280000, -260000, -240000, -220000, -200000, -180000, -160000], ['-3.2L', '-3.0L', '-2.8L', '-2.6L', '-2.4L', '-2.2L', '-2.0L', '-1.8L', '-1.6L'])
    plt.ylabel("Total Reward")
    plt.savefig("Plots/3dqn_reward_plot.png")
    plt.show()

    # Plot Waiting times v/s Episodes
    waiting_times = get_smoothed_val(waiting_times)
    plt.plot(episodes, waiting_times)
    plt.xlabel("Number of episodes")
    plt.ylabel("Average Waiting Time")
    plt.savefig("Plots/3dqn_waiting_time_plot.png")
    plt.show()

    # Plot epsilons v/s steps
    plt.plot(steps, epsilons)
    plt.xlabel("Number of steps")
    plt.ylabel("Epsilon value")
    plt.savefig("Plots/3dqn_epsilon_plot.png")
    plt.show()

    # Plot TD-error v/s Iterations
    x = iters
    y = get_smoothed_val(td_errors)
    y = clip_spikes(y, 0,0.008)
    plt.plot(x, y)
    plt.xlabel("Number of epochs")
    plt.ylabel("TD-Error")
    plt.savefig("Plots/3dqn_td_error_plot.png")
    plt.show()

