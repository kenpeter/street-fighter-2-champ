import argparse
import retro
import os
import time
import random
import math
from enum import Enum
import tensorflow as tf
from tqdm import tqdm
import logging
from DeepQAgent import DeepQAgent, Moves
import multiprocessing as mp
import numpy as np
from multiprocessing import Process, Queue, Manager
import threading
from queue import Queue as ThreadQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lobby.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Lobby")

# Ensure required directories exist
REQUIRED_DIRS = ["./models", "./logs"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    try:
        logger.info(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info("GPU memory growth enabled.")
        tf.config.set_visible_devices(physical_devices[0], "GPU")
        logger.info(f"Set visible GPU device: {physical_devices[0]}")
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.warning("No GPU found. Will use CPU instead.")


class Lobby_Full_Exception(Exception):
    pass


class Lobby_Modes(Enum):
    SINGLE_PLAYER = 1
    TWO_PLAYER = 2


def make_env_worker(env_id, game, state_name, result_queue, command_queue):
    """Worker process that runs a single environment"""
    try:
        # Each process gets its own environment
        env = retro.make(game=game, players=1)
        env.reset()

        # Load the state
        state_path = os.path.join(
            os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
            f"{state_name}.state",
        )
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state_data = f.read()
            env.em.set_state(state_data)

        # RAM addresses for reading game state
        ram_info = {
            "continue_timer": {"address": 16744917, "type": "|u1"},
            "round_timer": {"address": 16750378, "type": ">u2"},
            "enemy_health": {"address": 16745154, "type": ">i2"},
            "enemy_x_position": {"address": 16745094, "type": ">u2"},
            "enemy_y_position": {"address": 16745098, "type": ">u2"},
            "enemy_matches_won": {"address": 16745559, "type": ">u4"},
            "enemy_status": {"address": 16745090, "type": ">u2"},
            "enemy_character": {"address": 16745563, "type": "|u1"},
            "health": {"address": 16744514, "type": ">i2"},
            "x_position": {"address": 16744454, "type": ">u2"},
            "y_position": {"address": 16744458, "type": ">u2"},
            "status": {"address": 16744450, "type": ">u2"},
            "matches_won": {"address": 16744922, "type": "|u1"},
            "score": {"address": 16744936, "type": ">d4"},
        }

        def read_ram_values(info):
            try:
                if hasattr(env.unwrapped, "get_ram"):
                    ram = env.unwrapped.get_ram()
                elif hasattr(env.unwrapped, "em") and hasattr(
                    env.unwrapped.em, "get_ram"
                ):
                    ram = env.unwrapped.em.get_ram()
                else:
                    return ensure_required_keys(info)

                for key, address_info in ram_info.items():
                    addr = address_info["address"]
                    data_type = address_info["type"]
                    if addr >= len(ram):
                        continue
                    try:
                        if data_type == "|u1":
                            value = ram[addr]
                        elif data_type == ">u2":
                            if addr + 1 < len(ram):
                                value = (ram[addr] << 8) | ram[addr + 1]
                            else:
                                continue
                        elif data_type == ">i2":
                            if addr + 1 < len(ram):
                                value = (ram[addr] << 8) | ram[addr + 1]
                                if value >= 32768:
                                    value -= 65536
                            else:
                                continue
                        elif data_type == ">u4":
                            if addr + 3 < len(ram):
                                value = (
                                    (ram[addr] << 24)
                                    | (ram[addr + 1] << 16)
                                    | (ram[addr + 2] << 8)
                                    | ram[addr + 3]
                                )
                            else:
                                continue
                        elif data_type == ">d4":
                            if addr + 3 < len(ram):
                                import struct

                                try:
                                    value = struct.unpack(
                                        ">f",
                                        bytes(
                                            [
                                                ram[addr],
                                                ram[addr + 1],
                                                ram[addr + 2],
                                                ram[addr + 3],
                                            ]
                                        ),
                                    )[0]
                                except struct.error:
                                    value = 0
                            else:
                                continue
                        else:
                            value = ram[addr]
                        info[key] = value
                    except Exception as e:
                        pass
            except Exception as e:
                pass
            return ensure_required_keys(info)

        def ensure_required_keys(info):
            required_keys = {
                "continue_timer": 0,
                "round_timer": 0,
                "enemy_health": 176,
                "enemy_x_position": 200,
                "enemy_y_position": 0,
                "enemy_matches_won": 0,
                "enemy_status": 512,
                "enemy_character": 0,
                "health": 176,
                "x_position": 100,
                "y_position": 0,
                "status": 512,
                "matches_won": 0,
                "score": 0,
            }
            for key, default_value in required_keys.items():
                if key not in info:
                    info[key] = default_value
            return info

        # Initialize environment
        NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        step_result = env.step(NO_ACTION)
        if len(step_result) == 4:
            last_observation, reward, done, last_info = step_result
        else:
            last_observation, reward, terminated, truncated, last_info = step_result
            done = terminated or truncated

        last_info = read_ram_values(last_info)

        # Send ready signal
        result_queue.put(("ready", env_id, None))

        # Main loop - wait for commands
        while True:
            try:
                command = command_queue.get(timeout=1)
                if command[0] == "step":
                    action_index, frame_inputs = command[1], command[2]

                    # Execute frame inputs
                    total_reward = 0
                    final_info = None
                    final_obs = None

                    # Reward calculation variables
                    full_hp = 176
                    reward_coeff = 3.0
                    prev_player_health = last_info.get("health", full_hp)
                    prev_opponent_health = last_info.get("enemy_health", full_hp)

                    for frame in frame_inputs:
                        step_result = env.step(frame)
                        if len(step_result) == 4:
                            obs, _, done, info = step_result
                        else:
                            obs, _, terminated, truncated, info = step_result
                            done = terminated or truncated

                        info = read_ram_values(info)
                        final_info = info
                        final_obs = obs

                        if (
                            info.get("health", full_hp) <= 0
                            or info.get("enemy_health", full_hp) <= 0
                        ):
                            done = True
                            break

                    # Calculate custom reward
                    if final_info:
                        curr_player_health = final_info.get("health", full_hp)
                        curr_opponent_health = final_info.get("enemy_health", full_hp)

                        if curr_player_health < 0:
                            total_reward = -math.pow(
                                full_hp, (curr_opponent_health + 1) / (full_hp + 1)
                            )
                            done = True
                        elif curr_opponent_health < 0:
                            total_reward = (
                                math.pow(
                                    full_hp, (curr_player_health + 1) / (full_hp + 1)
                                )
                                * reward_coeff
                            )
                            done = True
                        else:
                            total_reward = reward_coeff * (
                                prev_opponent_health - curr_opponent_health
                            ) - (prev_player_health - curr_player_health)

                    # Send step result
                    step_data = {
                        "obs": final_obs,
                        "info": final_info,
                        "reward": total_reward,
                        "done": done,
                        "prev_obs": last_observation,
                        "prev_info": last_info,
                        "action": action_index,
                    }
                    result_queue.put(("step_result", env_id, step_data))

                    # Update for next step
                    last_observation = final_obs
                    last_info = final_info

                elif command[0] == "reset":
                    # Reset environment
                    env.reset()
                    if os.path.exists(state_path):
                        with open(state_path, "rb") as f:
                            state_data = f.read()
                        env.em.set_state(state_data)

                    step_result = env.step(NO_ACTION)
                    if len(step_result) == 4:
                        last_observation, reward, done, last_info = step_result
                    else:
                        last_observation, reward, terminated, truncated, last_info = (
                            step_result
                        )
                        done = terminated or truncated

                    last_info = read_ram_values(last_info)
                    result_queue.put(("reset_done", env_id, None))

                elif command[0] == "close":
                    break

            except:
                continue

        env.close()

    except Exception as e:
        result_queue.put(("error", env_id, str(e)))


class VectorizedLobby:
    """Vectorized lobby using multiple processes"""

    def __init__(
        self, game="StreetFighterIISpecialChampionEdition-Genesis", num_envs=16
    ):
        self.game = game
        self.num_envs = num_envs
        self.agent = None

        # Simple win/loss tracking
        self.wins = 0
        self.losses = 0

        # Initialize processes
        self.processes = []
        self.result_queues = []
        self.command_queues = []
        self.episode_data = [
            [] for _ in range(num_envs)
        ]  # Store experience for each env
        self.episode_rewards = [0.0 for _ in range(num_envs)]
        self.episode_steps = [0 for _ in range(num_envs)]

    def add_agent(self, agent):
        """Add the main agent"""
        self.agent = agent
        agent.lobby = self

    @staticmethod
    def getStates():
        directory = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return []
        try:
            files = os.listdir(directory)
            states = [
                os.path.splitext(file)[0] for file in files if file.endswith(".state")
            ]
            if not states:
                return []
            logger.info(f"Found states: {states}")
            return states
        except Exception as e:
            logger.error(f"Error getting states: {e}")
            return []

    def start_processes(self):
        """Start all environment processes"""
        states = self.getStates()
        if not states:
            logger.warning("No state files found. Creating a default state...")
            create_default_state()
            states = ["default"]

        logger.info(f"Starting {self.num_envs} environment processes...")

        for env_id in range(self.num_envs):
            result_queue = Queue()
            command_queue = Queue()

            state_name = random.choice(states)
            process = Process(
                target=make_env_worker,
                args=(env_id, self.game, state_name, result_queue, command_queue),
            )
            process.start()

            self.processes.append(process)
            self.result_queues.append(result_queue)
            self.command_queues.append(command_queue)

        # Wait for all environments to be ready
        ready_count = 0
        while ready_count < self.num_envs:
            for i, queue in enumerate(self.result_queues):
                try:
                    msg_type, env_id, data = queue.get(timeout=0.1)
                    if msg_type == "ready":
                        ready_count += 1
                        logger.info(f"Environment {env_id} ready")
                    elif msg_type == "error":
                        logger.error(f"Environment {env_id} error: {data}")
                except:
                    continue

        logger.info(f"All {self.num_envs} environments ready!")

    def run_training(self, target_timesteps):
        """Execute training until target timesteps reached"""
        start_time = time.time()

        # Start processes
        self.start_processes()

        logger.info(
            f"Running training for {target_timesteps} timesteps across {self.num_envs} parallel environments"
        )

        active_envs = set(range(self.num_envs))
        completed_episodes = 0

        # Progress bar based on timesteps
        pbar = tqdm(total=target_timesteps, desc="Training Timesteps")

        try:
            while self.agent.current_timesteps < target_timesteps and active_envs:
                # Get actions for all active environments
                for env_id in list(active_envs):
                    # Get current observation from last step (simplified for this example)
                    # In practice, you'd track the current state
                    dummy_obs = np.zeros((224, 256, 3))  # Placeholder
                    dummy_info = {
                        "health": 176,
                        "enemy_health": 176,
                        "x_position": 100,
                        "enemy_x_position": 200,
                    }

                    # Get action from agent
                    if len(physical_devices) > 0:
                        with tf.device("/GPU:0"):
                            action_index, frame_inputs = self.agent.getMove(
                                dummy_obs, dummy_info
                            )
                    else:
                        action_index, frame_inputs = self.agent.getMove(
                            dummy_obs, dummy_info
                        )

                    # Send step command
                    self.command_queues[env_id].put(
                        ("step", action_index, frame_inputs)
                    )

                # Collect results
                for env_id in list(active_envs):
                    try:
                        msg_type, returned_env_id, data = self.result_queues[
                            env_id
                        ].get(timeout=10)

                        if msg_type == "step_result":
                            # Process step result
                            self.episode_steps[env_id] += 1
                            self.episode_rewards[env_id] += data["reward"]

                            # Store experience
                            experience = (
                                data["prev_obs"],
                                data["prev_info"],
                                data["action"],
                                data["reward"],
                                data["obs"],
                                data["info"],
                                data["done"],
                            )
                            self.episode_data[env_id].append(experience)

                            # Check if episode is done
                            if data["done"] or self.episode_steps[env_id] >= 2500:
                                # Episode finished
                                completed_episodes += 1

                                # Record episode data
                                for exp in self.episode_data[env_id]:
                                    self.agent.recordStep(exp)

                                # Determine win/loss
                                final_player_health = data["info"].get("health", 0)
                                final_enemy_health = data["info"].get("enemy_health", 0)
                                if final_player_health > final_enemy_health:
                                    self.wins += 1
                                else:
                                    self.losses += 1

                                # Reset for next episode
                                self.episode_data[env_id] = []
                                self.episode_rewards[env_id] = 0.0
                                self.episode_steps[env_id] = 0

                                # Update progress bar
                                pbar.n = self.agent.current_timesteps
                                pbar.refresh()

                                # Check if we've reached target timesteps
                                if self.agent.current_timesteps >= target_timesteps:
                                    break

                                # Reset environment for next episode
                                self.command_queues[env_id].put(("reset",))

                    except Exception as e:
                        logger.error(f"Error processing environment {env_id}: {e}")
                        continue

        finally:
            # Close all processes
            pbar.close()
            self.close_processes()

        # Train the agent on all collected experiences
        logger.info("Training agent on collected experiences...")
        if len(physical_devices) > 0:
            with tf.device("/GPU:0"):
                self.agent.reviewFight()
        else:
            self.agent.reviewFight()

        # Print final results
        self.print_final_results(start_time)

    def close_processes(self):
        """Close all environment processes"""
        logger.info("Closing environment processes...")

        # Send close commands
        for command_queue in self.command_queues:
            try:
                command_queue.put(("close",))
            except:
                pass

        # Wait for processes to finish
        for process in self.processes:
            try:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            except:
                pass

        logger.info("All processes closed")

    def print_final_results(self, start_time):
        """Print final win rate"""
        total_time = time.time() - start_time
        total_games = self.wins + self.losses
        win_rate = (self.wins / total_games * 100) if total_games > 0 else 0

        print("\n" + "=" * 50)
        print("FINAL TRAINING RESULTS")
        print("=" * 50)
        print(f"Total Games: {total_games}")
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Timesteps: {self.agent.current_timesteps}")
        print(f"Training Time: {total_time:.2f} seconds")
        print("=" * 50)


def create_default_state():
    """Create a default state file"""
    state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "default.state")

    try:
        env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
        env.reset()

        for _ in range(10):
            env.step([0] * len(env.buttons))

        state_data = env.em.get_state()

        with open(state_path, "wb") as f:
            f.write(state_data)
        with open(os.path.join(state_dir, "default"), "wb") as f:
            f.write(state_data)

        logger.info(f"Created default state at {state_path}")
        env.close()
        return "default"
    except Exception as e:
        logger.error(f"Error creating default state: {e}")
        return None


if __name__ == "__main__":
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("GPU devices: %s", tf.config.list_physical_devices("GPU"))

    parser = argparse.ArgumentParser(
        description="Run the Street Fighter II AI training lobby with vectorized environments"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50000,
        help="Total number of timesteps to train for",
    )

    args = parser.parse_args()

    # Always create a default state if needed
    states = VectorizedLobby.getStates()
    if not states:
        logger.info("No state files found. Creating a default state...")
        create_default_state()

    # Create agent with timestep-based scheduling
    agent = DeepQAgent(stateSize=60, total_timesteps=args.total_timesteps)

    # Use vectorized training
    logger.info(
        f"Starting vectorized training for {args.total_timesteps} timesteps with 16 processes"
    )
    lobby = VectorizedLobby(num_envs=16)
    lobby.add_agent(agent)
    lobby.run_training(args.total_timesteps)
