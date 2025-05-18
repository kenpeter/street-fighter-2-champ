import argparse
import retro
import os
import time
import random
from enum import Enum
import tensorflow as tf
from tqdm import tqdm
import logging
from DeepQAgent import DeepQAgent, Moves

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lobby.log"),
        logging.StreamHandler()
    ]
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
        logger.info(f"Set visible GPU device: {physical_devices[0].name}")
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.warning("No GPU found. Will use CPU instead.")

class Lobby_Full_Exception(Exception):
    pass

class Lobby_Modes(Enum):
    SINGLE_PLAYER = 1
    TWO_PLAYER = 2

class Lobby:
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    FRAME_RATE = 1 / 60  # Align with 60 FPS for smoother rendering

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

    def __init__(
        self,
        game="StreetFighterIISpecialChampionEdition-Genesis",
        mode=Lobby_Modes.SINGLE_PLAYER,
    ):
        self.game = game
        self.mode = mode
        self.clearLobby()
        self.environment = None
        self.training_stats = {
            "episodes_run": 0,
            "total_steps": 0,
            "wins": 0,
            "losses": 0,
            "episode_rewards": [],
            "avg_training_loss": [],
            "session_start_time": time.time(),
            "session_wins": 0,
            "session_losses": 0,
        }
        self.ram_info = {
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

    def read_ram_values(self, info):
        if self.environment is None:
            return info
        try:
            if hasattr(self.environment.unwrapped, "get_ram"):
                ram = self.environment.unwrapped.get_ram()
            elif hasattr(self.environment.unwrapped, "em") and hasattr(
                self.environment.unwrapped.em, "get_ram"
            ):
                ram = self.environment.unwrapped.em.get_ram()
            else:
                return self.ensureRequiredKeys(info)
            for key, address_info in self.ram_info.items():
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
                    logger.error(f"Error reading RAM for {key}: {e}")
        except Exception as e:
            logger.error(f"Error reading RAM: {e}")
        return self.ensureRequiredKeys(info)

    def initEnvironment(self, state):
        logger.info(f"Initializing environment with state: {state}")
        try:
            if self.environment is not None:
                try:
                    self.environment.close()
                    logger.info("Closed existing environment")
                except Exception as close_error:
                    logger.warning(f"Warning when closing environment: {close_error}")
                self.environment = None
            
            self.environment = retro.make(game=self.game, players=self.mode.value)
            self.environment.reset()
            
            state_path = os.path.join(
                os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
                f"{state}.state",
            )
            if os.path.exists(state_path):
                logger.info(f"Loading state from: {state_path}")
                try:
                    with open(state_path, "rb") as f:
                        state_data = f.read()
                    self.environment.em.set_state(state_data)
                    logger.info(f"Loaded state successfully")
                except Exception as state_error:
                    logger.warning(f"Warning when loading state: {state_error}")
            
            step_result = self.environment.step(Lobby.NO_ACTION)
            if len(step_result) == 4:
                self.lastObservation, reward, done, self.lastInfo = step_result
                self.done = done
            else:
                self.lastObservation, reward, terminated, truncated, self.lastInfo = (
                    step_result
                )
                self.done = terminated or truncated
            
            logger.info("Environment stepped with NO_ACTION")
            logger.info(f"Info keys available: {list(self.lastInfo.keys())}")
            
            self.lastInfo = self.read_ram_values(self.lastInfo)
            logger.info(f"Info keys after RAM reading: {list(self.lastInfo.keys())}")
            
            self.lastAction, self.frameInputs = 0, [Lobby.NO_ACTION]
            self.episode_steps = 0
            self.episode_reward = 0
            self.initial_health = self.lastInfo.get("health", 100)
            self.initial_enemy_health = self.lastInfo.get("enemy_health", 100)
        
        except Exception as e:
            logger.error(f"Error initializing environment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def addPlayer(self, newPlayer):
        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return
        raise Lobby_Full_Exception(
            "Lobby has already reached the maximum number of players"
        )

    def clearLobby(self):
        self.players = [None] * self.mode.value

    def monitor_game_state(self, info, step_count):
        terminal_states = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]
        
        state_log = {
            "step": step_count,
            "health": info.get("health", -1),
            "enemy_health": info.get("enemy_health", -1),
            "status": info.get("status", -1),
            "enemy_status": info.get("enemy_status", -1),
            "x_position": info.get("x_position", -1),
            "enemy_x_position": info.get("enemy_x_position", -1),
            "done_flag": self.done,
        }
        
        if info.get("status", 0) in terminal_states:
            logger.warning(f"WARNING: Player in terminal state: {info['status']}")
        if info.get("enemy_status", 0) in terminal_states:
            logger.warning(f"WARNING: Enemy in terminal state: {info['enemy_status']}")
        if info.get("health", 100) <= 0:
            logger.warning(f"ALERT: Player health is zero or negative: {info['health']}")
        if info.get("enemy_health", 100) <= 0:
            logger.warning(f"ALERT: Enemy health is zero or negative: {info['enemy_health']}")
            
        unusual_event = (
            info.get("health", 100) <= 20
            or info.get("enemy_health", 100) <= 20
            or info.get("status", 0) in terminal_states
            or info.get("enemy_status", 0) in terminal_states
        )
        if step_count % 100 == 0 or unusual_event:
            logger.info(f"STATE LOG [{step_count}]: {state_log}")
            
        return state_log

    def play(self, state):
        try:
            self.initEnvironment(state)
            max_steps = 2500
            step_count = 0
            last_states = []
            
            while not self.done and step_count < max_steps:
                step_count += 1
                self.episode_steps += 1
                self.training_stats["total_steps"] += 1
                
                if len(physical_devices) > 0:
                    with tf.device("/GPU:0"):
                        self.lastAction, self.frameInputs = self.players[0].getMove(
                            self.lastObservation, self.lastInfo
                        )
                else:
                    self.lastAction, self.frameInputs = self.players[0].getMove(
                        self.lastObservation, self.lastInfo
                    )
                    
                try:
                    info, obs = self.enterFrameInputs()
                except Exception as e:
                    logger.error(f"ERROR during frame inputs: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error(f"Last known state before error: {self.lastInfo}")
                    self.done = True
                    break
                    
                state_log = self.monitor_game_state(info, step_count)
                if state_log:
                    last_states.append(state_log)
                    if len(last_states) > 5:
                        last_states.pop(0)
                        
                self.episode_reward += self.lastReward
                
                # when we play, we push last obs, state, action, reward, curr obs, into mem
                self.players[0].recordStep(
                    (
                        self.lastObservation,
                        self.lastInfo,
                        self.lastAction,
                        self.lastReward,
                        obs,
                        info,
                        self.done,
                    )
                )
                
                self.lastObservation, self.lastInfo = obs, info
                
                if step_count % 100 == 0:
                    logger.info(
                        f"Step {step_count}, Player health: {info['health']}, Enemy health: {info['enemy_health']}"
                    )
                
            self.training_stats["episodes_run"] += 1
            self.training_stats["episode_rewards"].append(self.episode_reward)
            
            if self.lastInfo.get("health", 0) > self.lastInfo.get("enemy_health", 0):
                self.training_stats["wins"] += 1
                self.training_stats["session_wins"] += 1
                logger.info("Episode result: WIN")
            else:
                self.training_stats["losses"] += 1
                self.training_stats["session_losses"] += 1
                logger.info("Episode result: LOSS")
                
            logger.info(f"Episode steps: {self.episode_steps}")
            logger.info(f"Episode reward: {self.episode_reward}")
            logger.info(f"Total steps so far: {self.training_stats['total_steps']}")
            
            if step_count >= max_steps:
                logger.warning(
                    "WARNING: Episode terminated due to step limit, not game completion"
                )
            else:
                logger.info("Episode completed naturally")
                
            if self.environment is not None:
                try:
                    self.environment.close()
                    self.environment = None
                except Exception as e:
                    logger.error(f"Error closing environment: {e}")
                    
        except Exception as e:
            logger.error(f"Error playing state {state}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.environment is not None:
                try:
                    self.environment.close()
                    self.environment = None
                except Exception as e:
                    logger.error(f"Error closing environment after exception: {e}")

    def ensureRequiredKeys(self, info):
        required_keys = {
            "continue_timer": 0,
            "round_timer": 0,
            "enemy_health": 100,
            "enemy_x_position": 200,
            "enemy_y_position": 0,
            "enemy_matches_won": 0,
            "enemy_status": 512,
            "enemy_character": 0,
            "health": 100,
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

    # this is the main reward func
    def enterFrameInputs(self):
        self.lastReward = 0
        start_time = time.time()
        
        # Store the initial state values for reward calculation
        initial_state = {
            "health": self.lastInfo.get("health", 100),
            "enemy_health": self.lastInfo.get("enemy_health", 100),
        }
        
        for frame in self.frameInputs:
            step_result = self.environment.step(frame)
            if len(step_result) == 4:
                obs, tempReward, self.done, info = step_result
            else:
                obs, tempReward, terminated, truncated, info = step_result
                self.done = terminated or truncated
                    
            info = self.read_ram_values(info)
            
            # Calculate custom rewards based on changes from initial state to current state
            # Reward for damage dealt to opponent
            damage_dealt = max(0, initial_state["enemy_health"] - info.get("enemy_health", 100))
            damage_reward = damage_dealt * 0.3  # Scale factor can be tuned
            
            # Penalty for damage taken
            damage_taken = max(0, initial_state["health"] - info.get("health", 100))
            defense_reward = -damage_taken * 0.15  # Slightly higher penalty for taking damage
            
            # Small reward for health advantage
            health_diff = info.get("health", 100) - info.get("enemy_health", 100)
            health_diff_reward = health_diff * 0.01  # Small reward for health advantage
            
            # Combine all reward components
            custom_reward = damage_reward + defense_reward + health_diff_reward
            
            # Add significant win/loss rewards
            if info.get("enemy_health", 100) <= 0:  # Win condition
                custom_reward += 20.0  # Large positive reward for winning
            elif info.get("health", 100) <= 0:  # Loss condition
                custom_reward -= 15.0  # Large negative reward for losing
                
            # Add custom reward to environment reward
            self.lastReward += tempReward + custom_reward
            
            if info.get("health", 100) <= 0 or info.get("enemy_health", 100) <= 0:
                self.done = True
                logger.info(
                    f"Game terminated: Player health={info.get('health', 'Unknown')}, Enemy health={info.get('enemy_health', 'Unknown')}"
                )
                    
            if self.done:
                frame_time = (time.time() - start_time) * 1000  # ms
                logger.debug(f"Frame processing time: {frame_time:.2f}ms")
                logger.debug(f"Final reward: {self.lastReward} (includes custom reward: {custom_reward})")
                return info, obs
                    
        frame_time = (time.time() - start_time) * 1000  # ms
        logger.debug(f"Frame processing time: {frame_time:.2f}ms")
        return info, obs


    # we speicify episode, then pass donw here
    def executeTrainingRun(self, review=True, episodes=1):
        start_time = time.time()
        self.training_stats["session_wins"] = 0
        self.training_stats["session_losses"] = 0
        self.training_stats["session_start_time"] = time.time()

        for episodeNumber in tqdm(range(episodes), desc="Training Episodes"):
            logger.info(f"\n=== Starting episode {episodeNumber+1}/{episodes} ===")
            
            states = Lobby.getStates()
            if not states:
                logger.warning("No state files found. Creating a default state...")
                try:
                    create_default_state()
                    states = ["default"]
                except Exception as e:
                    logger.error(f"Error creating default state: {e}")
                    logger.error(
                        "Please create at least one state file before running training."
                    )
                    return
                    
            self.episode_steps = 0
            self.episode_reward = 0
            
            # we load the state file, fight each enemy, then we play (inject all info into mem)
            # finish play, we review it (train it)
            for state in states:
                logger.info(f"Loading state: {state}")
                try:
                    self.play(state=state)
                    
                    if review and self.players[0].__class__.__name__ != "Agent":
                        if len(physical_devices) > 0:
                            with tf.device("/GPU:0"):
                                self.players[0].reviewFight()
                        else:
                            self.players[0].reviewFight()
                except Exception as e:
                    logger.error(f"Error playing state {state}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    if self.environment is not None:
                        try:
                            self.environment.close()
                            self.environment = None
                        except:
                            pass
                    continue

        total_time = time.time() - start_time
        win_rate = (
            (self.training_stats["wins"] / self.training_stats["episodes_run"]) * 100
            if self.training_stats["episodes_run"] > 0
            else 0
        )
        logger.info("\n========= TRAINING SESSION SUMMARY =========")
        logger.info(f"Total training steps: {self.training_stats['total_steps']}")
        logger.info(f"Total episodes: {self.training_stats['episodes_run']}")
        logger.info(
            f"Total Win/Loss Record: {self.training_stats['wins']}W - {self.training_stats['losses']}L ({win_rate:.2f}%)"
        )
        
        session_win_rate = (
            (
                self.training_stats["session_wins"]
                / (self.training_stats["session_losses"] + self.training_stats["session_wins"])
            )
            * 100
            if (self.training_stats["session_losses"] + self.training_stats["session_wins"]) > 0
            else 0
        )
        logger.info(
            f"Current session record: {self.training_stats['session_wins']}W - {self.training_stats['session_losses']}L ({session_win_rate:.2f}%)"
        )
        
        if hasattr(self.players[0], "loaded_stats") and self.players[0].loaded_stats:
            previous_wins = (
                self.training_stats["wins"] - self.training_stats["session_wins"]
            )
            previous_losses = (
                self.training_stats["losses"] - self.training_stats["session_losses"]
            )
            previous_episodes = previous_wins + previous_losses
            if previous_episodes > 0:
                previous_win_rate = (previous_wins / previous_episodes) * 100
                win_rate_change = session_win_rate - previous_win_rate
                logger.info(
                    f"Win rate change: {win_rate_change:+.2f}% (Previous: {previous_win_rate:.2f}%)"
                )
                if win_rate_change > 5:
                    logger.info("Performance trend: STRONG IMPROVEMENT")
                elif win_rate_change > 0:
                    logger.info("Performance trend: SLIGHT IMPROVEMENT")
                elif win_rate_change > -5:
                    logger.info("Performance trend: STABLE")
                else:
                    logger.info("Performance trend: DECLINING")
                    
        accumulated_stats = "No"
        if hasattr(self.players[0], "total_timesteps"):
            if (
                hasattr(self.players[0], "loaded_stats")
                and self.players[0].loaded_stats
            ):
                accumulated_stats = "Yes (--resume)"
        logger.info(f"Accumulated stats: {accumulated_stats}")
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
        steps_per_second = (
            self.training_stats["total_steps"] / total_time if total_time > 0 else 0
        )
        logger.info(f"Training efficiency: {steps_per_second:.2f} steps/second")
        
        if hasattr(self.players[0], "total_timesteps"):
            logger.info(
                f"Agent's accumulated training timesteps: {self.players[0].total_timesteps}"
            )
            
        if len(self.training_stats["episode_rewards"]) >= 2:
            first_episodes = self.training_stats["episode_rewards"][
                : min(3, len(self.training_stats["episode_rewards"]))
            ]
            last_episodes = self.training_stats["episode_rewards"][
                -min(3, len(self.training_stats["episode_rewards"])) :
            ]
            first_rewards = (
                sum(first_episodes) / len(first_episodes) if first_episodes else 0
            )
            last_rewards = (
                sum(last_episodes) / len(last_episodes) if last_episodes else 0
            )
            if abs(first_rewards) > 0.001:
                reward_percent_change = (
                    (last_rewards - first_rewards) / abs(first_rewards)
                ) * 100
                logger.info(f"Reward trend: {reward_percent_change:+.2f}% change")
            else:
                reward_trend = last_rewards - first_rewards
                logger.info(f"Reward trend: {reward_trend:+.2f}")
                
            if last_rewards > first_rewards:
                logger.info("Learning assessment: POSITIVE - Agent is improving")
            elif last_rewards > first_rewards * 0.95:
                logger.info("Learning assessment: NEUTRAL - Agent performance is stable")
            else:
                logger.warning(
                    "Learning assessment: NEGATIVE - Agent may be stuck in suboptimal policy"
                )
        logger.info("===========================================")

def create_default_state():
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
        description="Run the Street Fighter II AI training lobby with CUDA support"
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=10,
        help="Integer representing the number of training rounds to go through, checkpoints are made at the end of each episode",
    )
    parser.add_argument(
        "-re",
        "--resume",
        action="store_true",
        help="Boolean flag for loading a pre-existing model and stats with higher exploration for continued training",
    )
    args = parser.parse_args()
    
    # Always create a default state if needed
    states = Lobby.getStates()
    if not states:
        logger.info("No state files found. Creating a default state...")
        create_default_state()
    
    # Always use GPU if available and always show progress
    lobby = Lobby()
    agent = DeepQAgent(
        stateSize=40,
        resume=args.resume,
        lobby=lobby,
    )
    
    lobby.addPlayer(agent)
    lobby.executeTrainingRun(episodes=args.episodes)