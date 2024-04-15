# ---------------------------------------------------------IMPORTS------------------------------------------------------
import frozen_lake_alts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import gym

from typing import NamedTuple
from pathlib import Path
from tqdm import tqdm
from frozen_lake_alts.envs.alt_reward import generate_random_map, generate_random_map_any_start


# ---------------------------------------------------------CLASSES------------------------------------------------------
class Params(NamedTuple):
    episodes: int  # episodes per run
    alpha_: float
    gamma_: float
    epsilon_: float
    size: int  # amount of tiles s.t. the square board is [size X size]
    seed: int  # reproducible rng
    actions: int
    states: int
    p: float  # tile freezing probability for map generation
    runs: int  # total number of runs (stochasticity)
    start: str
    generator: list
    path: Path


class QLearning:
    def __init__(self, alpha_, gamma_, states, actions):
        self.qtable = None
        self.states = states
        self.actions = actions
        self.alpha_ = alpha_
        self.gamma_ = gamma_
        self.reset()

    def update(self, state, action, reward, new_state):
        """update the qtable using given states and actions according to the qlearning update model"""
        q_update = \
            (
                    self.qtable[state, action]
                    + (
                            self.alpha_
                            * (
                                    reward
                                    + self.gamma_
                                    * np.max(self.qtable[new_state, :])
                                    - self.qtable[state, action]
                            )
                    )
            )
        return q_update

    def reset(self):
        """resets the qtable values to zeroes"""
        self.qtable = np.zeros((self.states, self.actions))


class SARSA:
    def __init__(self, alpha_, gamma_, states, actions):
        self.qtable = None
        self.states = states
        self.actions = actions
        self.alpha_ = alpha_
        self.gamma_ = gamma_
        self.reset()

    def update(self, state, action, reward, new_state, new_action):
        """update the qtable using given states and actions according to the sarsa update model"""
        q_update = \
            (
                    self.qtable[state, action]
                    + (
                            self.alpha_
                            * (
                                    reward
                                    + (
                                            self.gamma_
                                            * self.qtable[new_state, new_action]
                                    )
                                    - self.qtable[state, action]
                            )
                    )
            )
        return q_update

    def reset(self):
        """resets the qtable values to zeroes"""
        self.qtable = np.zeros((self.states, self.actions))


class EpsilonGreedy:
    def __init__(self, epsilon_):
        self.epsilon_ = epsilon_

    def return_action(self, action_space, state, qtable):
        """typical epsilon greedy, but with added convention to ensure random direction choices upon qtable ties"""
        if np.random.random() < self.epsilon_:
            action = action_space.sample()

        else:
            if np.all(qtable[state, :]) == qtable[state, 0]:  # if all values in the state are the same
                action = action_space.sample()  # randomly choose direction (don't just always choose left)
            else:
                action = np.argmax(qtable[state, :])
        return action


# ---------------------------------------------------------DEFS---------------------------------------------------------
def run_env_q():
    # init
    rewards = np.zeros((params.episodes, params.runs))
    steps = np.zeros((params.episodes, params.runs))
    episodes = np.arange(params.episodes)
    qtables = np.zeros((params.runs, params.states, params.actions))
    states = []
    actions = []

    for run in range(params.runs):
        agent_q.reset()  # zero out qtable each run (data is stored for postprocessing elsewhere)
        for episode in tqdm(episodes, leave=False):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0
            step_penalty = 1

            while not done:
                action = policy.return_action(
                    action_space=env.action_space, state=state, qtable=agent_q.qtable
                )

                # log state and action pairs
                states.append(state)
                actions.append(action)

                # get data
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # penalize agent
                step_penalty -= 0.002
                reward = round(reward * step_penalty, 3)

                # update qtable
                agent_q.qtable[state, action] = agent_q.update(state, action, reward, new_state)

                # more data registration
                total_rewards += reward
                step += 1
                state = new_state

            # save rewards and steps for this episode
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        # save qtable for run before resetting for stochasticity
        qtables[run, :, :] = agent_q.qtable

    return rewards, steps, qtables, states, actions


def run_env_sarsa():
    # init
    rewards = np.zeros((params.episodes, params.runs))
    steps = np.zeros((params.episodes, params.runs))
    episodes = np.arange(params.episodes)
    qtables = np.zeros((params.runs, params.states, params.actions))
    states = []
    actions = []

    for run in range(params.runs):
        agent_sarsa.reset()  # zero out qtable each run (data is stored for postprocessing elsewhere)
        for episode in tqdm(episodes, leave=False):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0
            step_penalty = 1

            while not done:
                action = policy.return_action(
                    action_space=env.action_space,
                    state=state,
                    qtable=agent_sarsa.qtable
                )

                # log state and action pairs
                states.append(state)
                actions.append(action)

                # get data
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                new_action = policy.return_action(
                    action_space=env.action_space,
                    state=new_state,
                    qtable=agent_sarsa.qtable
                )

                # penalize agent
                step_penalty -= 0.002
                reward = round(reward * step_penalty, 3)

                # update qtable
                agent_sarsa.qtable[state, action] = agent_sarsa.update(
                    state, action, reward, new_state, new_action
                )

                # more data registration
                total_rewards += reward
                step += 1
                state = new_state

            # save rewards and steps for this episode
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        # save qtable for run before resetting for stochasticity
        qtables[run, :, :] = agent_sarsa.qtable

    return rewards, steps, qtables, states, actions


def postprocess(episodes, rewards, steps, map_size):
    """abstract simulated results into a format suitable for processing/visualizing"""
    cumulative_reward = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.runs),
            "Rewards": rewards.flatten,
            "Steps": steps.flatten,
        }
    )
    cumulative_reward["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    cumulative_reward["map_size"] = np.repeat(f"{map_size}x{map_size}", cumulative_reward.shape[0])

    step_count = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    step_count["map_size"] = np.repeat(f"{map_size}x{map_size}", step_count.shape[0])
    return cumulative_reward, step_count


def qtable_dir_map(qtable, map_size):
    """arrow mapping """
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_arrows(qtable_q, qtable_sarsa, map_size):
    qtable_q_maxval, qtable_q_dir = qtable_dir_map(qtable_q, map_size)
    qtable_sarsa_maxval, qtable_sarsa_dir = qtable_dir_map(qtable_sarsa, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))

    seaborn.heatmap(
        qtable_q_maxval,
        annot=qtable_q_dir,
        fmt="",
        ax=ax[0],
        cmap=seaborn.color_palette("rocket_r", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Directions Q-learning agent would take\n in given state")
    for _, spine in ax[0].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    # SARSA arrows
    seaborn.heatmap(
        qtable_sarsa_maxval,
        annot=qtable_sarsa_dir,
        fmt="",
        ax=ax[1],
        cmap=seaborn.color_palette("mako_r", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Directions SARSA agent would take\n in given state")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    # save figure
    img_title = f"arrowmap_def_{map_size}x{map_size}.png"
    fig.savefig(params.path / img_title, bbox_inches="tight")

    plt.show()


def plot_actions_hist(actions_q, actions_sarsa, map_size):
    """state-action distribution histogram"""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Q-learning actions
    seaborn.histplot(data=actions_q, ax=ax[0]).set(title="Q-learning actions")
    ax[0].set_xticks(list(labels.values()), labels=labels.keys())
    fig.tight_layout()

    # SARSA actions
    seaborn.histplot(data=actions_sarsa, ax=ax[1]).set(title="SARSA actions")
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    fig.tight_layout()

    # save figure
    img_title = f"actionshist_def_{map_size}x{map_size}.png"
    fig.savefig(params.path / img_title, bbox_inches="tight")

    plt.show()


def plot_steps_and_rewards(rewards_df, steps_df, title=None, palette="tab10"):
    labels = steps_df.map_size.unique()
    colors = seaborn.color_palette(palette=palette)

    new_palette = dict((labels[n], colors[n]) for n in range(len(labels)))

    fig = plt.figure(layout='constrained', figsize=(15, 5))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.05)

    left_ax = subfigs[0].subplots(nrows=1, ncols=1)
    seaborn.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=left_ax, palette=new_palette
    )
    subfigs[0].suptitle("Cumulated rewards")

    right_ax = subfigs[1].subplots(nrows=2, ncols=2)
    for count in range(4):
        df = steps_df[steps_df["map_size"] == labels[count]]
        if count >= 4:
            break

        seaborn.lineplot(
            data=df,
            x="Episodes",
            y="Steps",
            hue="map_size",
            ax=right_ax[int(count / 2)][count % 2],
            palette=new_palette
        )
        right_ax[int(count / 2)][count % 2].legend(title="map size")
    subfigs[1].suptitle("Averaged steps number")

    img_title = "default.png"

    if title is not None:
        fig.suptitle(title)
        img_title = f"{title}.png"

    fig.savefig(params.path / img_title, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------PROGRAM------------------------------------------------------
seaborn.set_theme()

# set general parameters
# (see ./frozen-lake-alts/frozen_lake_alts/envs/alt_reward.py for explicit declarations of p, start, and generator)
params = Params(
    episodes=20000,
    alpha_=0.8,
    gamma_=0.9,
    epsilon_=0.1,
    size=25,
    seed=112358,
    actions=None,
    states=None,
    p=0.85,
    runs=10,
    start="random",
    generator=[generate_random_map_any_start],
    path=Path("./training_data/img"),
)

# create img folder if not exists
params.path.mkdir(parents=True, exist_ok=True)

# dataframes for both types of learning
res_all_q = pd.DataFrame()
res_all_sarsa = pd.DataFrame()
st_all_q = pd.DataFrame()
st_all_sarsa = pd.DataFrame()

# map sizes
sizes = [9, 11, 15, 25]
for size in sizes:
    env = gym.make(
        "FrozenLakeAltReward",
        is_slippery=False,
        desc=generate_random_map(size, params.p)
    )

    # prepare global parameter usage
    params = params._replace(actions=env.action_space.n)
    params = params._replace(states=env.observation_space.n)
    params = params._replace(size=size)

    env.action_space.seed(params.seed)

    # initialize learning agents
    agent_q = QLearning(
        alpha_=params.alpha_,
        gamma_=params.gamma_,
        states=params.states,
        actions=params.actions
    )
    agent_sarsa = SARSA(
        alpha_=params.alpha_,
        gamma_=params.gamma_,
        states=params.states,
        actions=params.actions
    )
    policy = EpsilonGreedy(
        epsilon_=params.epsilon_
    )

    # run both environments
    rewards_q, steps_q, qtables_q, states_q, actions_q = run_env_q()
    rewards_sarsa, steps_sarsa, qtables_sarsa, states_sarsa, actions_sarsa = run_env_sarsa()

    # postprocessing for both environments to prepare for data visualization
    res, st = postprocess(np.arange(params.episodes), rewards_q, steps_q, size)
    res_all_q = pd.concat([res_all_q, res])
    st_all_q = pd.concat([st_all_q, st])
    qtable_q = qtables_q.mean(axis=0)

    res_sarsa, st_sarsa = postprocess(np.arange(params.episodes), rewards_sarsa, steps_sarsa, size)
    res_all_sarsa = pd.concat([res_all_sarsa, res_sarsa])
    st_all_sarsa = pd.concat([st_all_sarsa, st_sarsa])
    qtable_sarsa = qtables_sarsa.mean(axis=0)

    # data visualization
    plot_actions_hist(actions_q, actions_sarsa, size)
    plot_q_arrows(qtable_q, qtable_sarsa, size)

    env.close()

# more data visualization
plot_steps_and_rewards(res_all_q, st_all_q, "Q-Learning Rewards and Steps Default", 'rocket_r')
plot_steps_and_rewards(res_all_sarsa, st_all_sarsa, "SARSA Rewards and Steps Default", 'mako_r')
