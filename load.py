
from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

algorithm_map = {
    # TODO: maybe network architecture needs complication.
    "ppo": PPO,
    "a2c": A2C,
    "td3": TD3
}


model = TD3.load("./test.zip", env=env)