from gym.envs.registration import register

register(
    id="FrozenLakeAltReward",
    entry_point="frozen_lake_alts.envs.alt_reward:FrozenLakeAltRewardEnv",
    kwargs={"size": 4},
    max_episode_steps=500,
    reward_threshold=3,  # optimum = 0.74
)

register(
    id="FrozenLakeAltReward8x8",
    entry_point="frozen_lake_alts.envs.alt_reward:FrozenLakeAltRewardEnv",
    kwargs={"size": 8},
    max_episode_steps=200,
    reward_threshold=7,  # optimum = 0.91
)

register(
    id="FrozenLakeAltReward16x16",
    entry_point="frozen_lake_alts.envs.alt_reward:FrozenLakeAltRewardEnv",
    kwargs={"size": 16},
    max_episode_steps=400,
    reward_threshold=15,  # optimum = 0.91
)