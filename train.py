import os
import sys
import argparse

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import StreetFighterCustomWrapper

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(game, state, seed=0, rendering=False):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )
        env = StreetFighterCustomWrapper(env, rendering=rendering)
        env = Monitor(env)
        env.seed(seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Street Fighter")
    parser.add_argument(
        "--render", action="store_true", help="Render the game UI during training"
    )
    args = parser.parse_args()

    # Configure number of environments and vectorized environment type
    if args.render:
        num_env = 1
        rendering = True
        VecEnv = DummyVecEnv
    else:
        num_env = 16
        rendering = False
        VecEnv = SubprocVecEnv

    # Set up the environment
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = VecEnv(
        [
            make_env(
                game, state="Champion.Level12.RyuVsBison", seed=i, rendering=rendering
            )
            for i in range(num_env)
        ]
    )

    # Define schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Initialize the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
    )

    # Set up save directory and callbacks
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_interval = 31250  # Steps per checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_ryu"
    )

    # Redirect stdout to a log file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        sys.stdout = log_file
        model.learn(total_timesteps=int(100000000), callback=[checkpoint_callback])
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_sf2_ryu_final.zip"))


if __name__ == "__main__":
    main()
