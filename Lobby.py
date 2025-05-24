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
REQUIRED_DIRS = ["./models", "./logs", "./stats"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {os.path.abspath(directory)}")

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
    """Vectorized lobby using multiple processes like SubprocVecEnv"""

    def __init__(
        self, game="StreetFighterIISpecialChampionEdition-Genesis", num_envs=16
    ):
        self.game = game
        self.num_envs = num_envs
        self.agent = None
        self.training_stats = {
            "episodes_run": 0,
            "total_steps": 0,
            "wins": 0,
            "losses": 0,
            "episode_rewards": [],
            "session_start_time": time.time(),
            "session_wins": 0,
            "session_losses": 0,
        }

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

    def execute_parallel_training(self, episodes=10):
        """Execute training with vectorized environments"""
        start_time = time.time()
        self.training_stats["session_wins"] = 0
        self.training_stats["session_losses"] = 0
        self.training_stats["session_start_time"] = time.time()

        # Start processes
        self.start_processes()

        logger.info(
            f"Running {episodes} episodes across {self.num_envs} parallel environments"
        )

        episodes_per_env = episodes // self.num_envs
        remaining_episodes = episodes % self.num_envs

        # Track episodes per environment
        env_episodes_left = [
            episodes_per_env + (1 if i < remaining_episodes else 0)
            for i in range(self.num_envs)
        ]
        active_envs = set(range(self.num_envs))
        completed_episodes = 0

        # Progress bar
        pbar = tqdm(total=episodes, desc="Training Episodes")

        try:
            while active_envs and completed_episodes < episodes:
                # Get actions for all active environments
                for env_id in list(active_envs):
                    if env_episodes_left[env_id] > 0:
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
                                env_episodes_left[env_id] -= 1
                                completed_episodes += 1

                                # Record episode data
                                for exp in self.episode_data[env_id]:
                                    self.agent.recordStep(exp)

                                # Update stats
                                self.training_stats["episodes_run"] += 1
                                self.training_stats[
                                    "total_steps"
                                ] += self.episode_steps[env_id]
                                self.training_stats["episode_rewards"].append(
                                    self.episode_rewards[env_id]
                                )

                                # Determine win/loss
                                final_player_health = data["info"].get("health", 0)
                                final_enemy_health = data["info"].get("enemy_health", 0)
                                if final_player_health > final_enemy_health:
                                    self.training_stats["wins"] += 1
                                    self.training_stats["session_wins"] += 1
                                else:
                                    self.training_stats["losses"] += 1
                                    self.training_stats["session_losses"] += 1

                                # Reset for next episode
                                self.episode_data[env_id] = []
                                self.episode_rewards[env_id] = 0.0
                                self.episode_steps[env_id] = 0

                                pbar.update(1)

                                # Reset environment if more episodes needed
                                if env_episodes_left[env_id] > 0:
                                    self.command_queues[env_id].put(("reset",))
                                else:
                                    active_envs.discard(env_id)

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

        # Print training summary
        self.print_training_summary(start_time)

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

    def print_training_summary(self, start_time):
        """Print comprehensive training summary"""
        total_time = time.time() - start_time
        win_rate = (
            (self.training_stats["wins"] / self.training_stats["episodes_run"]) * 100
            if self.training_stats["episodes_run"] > 0
            else 0
        )

        logger.info("\n========= VECTORIZED TRAINING SESSION SUMMARY =========")
        logger.info(f"Parallel environments used: {self.num_envs}")
        logger.info(f"Total training steps: {self.training_stats['total_steps']}")
        logger.info(f"Total episodes: {self.training_stats['episodes_run']}")
        logger.info(
            f"Total Win/Loss Record: {self.training_stats['wins']}W - {self.training_stats['losses']}L ({win_rate:.2f}%)"
        )

        if hasattr(self.agent, "epsilon"):
            logger.info(f"Current epsilon value: {self.agent.epsilon:.4f}")

        session_win_rate = (
            (
                self.training_stats["session_wins"]
                / (
                    self.training_stats["session_losses"]
                    + self.training_stats["session_wins"]
                )
            )
            * 100
            if (
                self.training_stats["session_losses"]
                + self.training_stats["session_wins"]
            )
            > 0
            else 0
        )
        logger.info(
            f"Current session record: {self.training_stats['session_wins']}W - {self.training_stats['session_losses']}L ({session_win_rate:.2f}%)"
        )

        logger.info(f"Total training time: {total_time:.2f} seconds")

        steps_per_second = (
            self.training_stats["total_steps"] / total_time if total_time > 0 else 0
        )
        logger.info(f"Training efficiency: {steps_per_second:.2f} steps/second")

        episodes_per_second = (
            self.training_stats["episodes_run"] / total_time if total_time > 0 else 0
        )
        logger.info(f"Episode throughput: {episodes_per_second:.2f} episodes/second")

        if hasattr(self.agent, "total_timesteps"):
            logger.info(
                f"Agent's accumulated training timesteps: {self.agent.total_timesteps}"
            )

        logger.info("=====================================================")


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
        description="Run the Street Fighter II AI training lobby with 16 vectorized environments"
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=160,
        help="Total number of episodes to run across all parallel environments",
    )
    parser.add_argument(
        "-re",
        "--resume",
        action="store_true",
        help="Boolean flag for loading a pre-existing model and stats",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Exploration rate (epsilon) for the agent (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--rl",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Always create a default state if needed
    states = VectorizedLobby.getStates()
    if not states:
        logger.info("No state files found. Creating a default state...")
        create_default_state()

    # Create agent
    agent = DeepQAgent(stateSize=60, resume=args.resume)

    # Use vectorized training like SubprocVecEnv
    logger.info("Starting vectorized training with 16 separate processes")
    lobby = VectorizedLobby(num_envs=16)
    lobby.add_agent(agent)
    lobby.execute_parallel_training(episodes=args.episodes)
