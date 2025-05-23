import argparse
import retro
import os
import time
import random
from enum import Enum
import sys
import tensorflow as tf
from Discretizer import StreetFighter2Discretizer
from myagent import DeepQAgent  # Import the fixed DeepQAgent
from tqdm import tqdm
import logging

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
    NO_ACTION = 0
    MOVEMENT_BUTTONS = ["LEFT", "RIGHT", "DOWN", "UP"]
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]
    ROUND_TIMER_NOT_STARTED = 39208
    STANDING_STATUS = 512
    CROUCHING_STATUS = 514
    JUMPING_STATUS = 516
    ACTIONABLE_STATUSES = [STANDING_STATUS, CROUCHING_STATUS, JUMPING_STATUS]
    JUMP_LAG = 4
    FRAME_RATE = 1 / 300

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
        render=False,
        mode=Lobby_Modes.SINGLE_PLAYER,
    ):
        self.game = game
        self.render = render
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
            
            # Create and configure the environment
            self.environment = retro.make(game=self.game, players=self.mode.value)
            self.environment.reset()
            
            # Load state file
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
            
            # Apply discretizer for action space
            self.environment = StreetFighter2Discretizer(self.environment)
            logger.info("Applied StreetFighter2Discretizer")
            
            # Take an initial step to initialize game state
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
            
            # Read additional values from RAM
            self.lastInfo = self.read_ram_values(self.lastInfo)
            logger.info(f"Info keys after RAM reading: {list(self.lastInfo.keys())}")
            
            # Initialize episode state
            self.lastAction, self.frameInputs = 0, [Lobby.NO_ACTION]
            self.currentJumpFrame = 0
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

    def isActionableState(self, info, action=0):
        """Determine if an action can be taken based on game state"""
        action = self.environment.get_action_meaning(action)
        
        # Check if either player is dead
        if info.get("health", 100) <= 0:
            return False
        if info.get("enemy_health", 100) <= 0:
            return False
        
        # Check round timer
        if "round_timer" in info and info["round_timer"] == 0:
            return False
        if info["round_timer"] == Lobby.ROUND_TIMER_NOT_STARTED:
            return False
            
        # Handle jumping state
        elif (
            info["status"] == Lobby.JUMPING_STATUS
            and self.currentJumpFrame <= Lobby.JUMP_LAG
        ):
            self.currentJumpFrame += 1
            return False
        elif info["status"] == Lobby.JUMPING_STATUS and any(
            [button in action for button in Lobby.ACTION_BUTTONS]
        ):
            return False
            
        # Check if character is in actionable state
        elif info["status"] not in Lobby.ACTIONABLE_STATUSES:
            return False
            
        # All checks passed, state is actionable
        else:
            if info["status"] != Lobby.JUMPING_STATUS and self.currentJumpFrame > 0:
                self.currentJumpFrame = 0
            return True

    def monitor_game_state(self, info, step_count):
        """Log game state for debugging"""
        terminal_states = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]
        
        # Create a snapshot of current state
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
        
        # Log warnings for important game states
        if info.get("status", 0) in terminal_states:
            logger.warning(f"WARNING: Player in terminal state: {info['status']}")
        if info.get("enemy_status", 0) in terminal_states:
            logger.warning(f"WARNING: Enemy in terminal state: {info['enemy_status']}")
        if info.get("health", 100) <= 0:
            logger.warning(f"ALERT: Player health is zero or negative: {info['health']}")
        if info.get("enemy_health", 100) <= 0:
            logger.warning(f"ALERT: Enemy health is zero or negative: {info['enemy_health']}")
            
        # Log state periodically or during unusual events
        unusual_event = (
            info.get("health", 100) <= 20
            or info.get("enemy_health", 100) <= 20
            or info.get("status", 0) in terminal_states
            or info.get("enemy_status", 0) in terminal_states
        )
        if step_count % 100 == 0 or unusual_event:
            logger.info(f"STATE LOG [{step_count}]: {state_log}")
            
        return state_log

    # we actually play the game
    def play(self, state):
        """Main gameplay loop for an episode"""
        try:
            self.initEnvironment(state)
            max_steps = 2500
            step_count = 0
            last_states = []
            
            # Main game loop
            while not self.done and step_count < max_steps:
                step_count += 1
                self.episode_steps += 1
                self.training_stats["total_steps"] += 1
                
                # Get next action from agent, using GPU if available
                if len(physical_devices) > 0:
                    with tf.device("/GPU:0"):
                        self.lastAction, self.frameInputs = self.players[0].getMove(
                            self.lastObservation, self.lastInfo
                        )
                else:
                    self.lastAction, self.frameInputs = self.players[0].getMove(
                        self.lastObservation, self.lastInfo
                    )
                    
                # Apply action and get new state
                self.lastReward = 0
                try:
                    info, obs = self.enterFrameInputs()
                except Exception as e:
                    logger.error(f"ERROR during frame inputs: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error(f"Last known state before error: {self.lastInfo}")
                    self.done = True
                    break
                    
                # Monitor and log game state
                state_log = self.monitor_game_state(info, step_count)
                if state_log:
                    last_states.append(state_log)
                    if len(last_states) > 5:
                        last_states.pop(0)
                        
                # Update episode metrics
                self.episode_reward += self.lastReward
                
                # Record experience
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
                
                # Update current state
                self.lastObservation, self.lastInfo = obs, info
                
                # Periodic logging
                if step_count % 100 == 0:
                    logger.info(
                        f"Step {step_count}, Player health: {info['health']}, Enemy health: {info['enemy_health']}"
                    )
                    
            # Episode ended unexpectedly - log details
            if self.done and step_count < max_steps:
                logger.warning("\n===== GAME CLOSED UNEXPECTEDLY =====")
                logger.warning(f"Steps completed: {step_count}/{max_steps}")
                logger.warning(
                    f"Last known player health: {self.lastInfo.get('health', 'Unknown')}"
                )
                logger.warning(
                    f"Last known enemy health: {self.lastInfo.get('enemy_health', 'Unknown')}"
                )
                logger.warning(
                    f"Last known player status: {self.lastInfo.get('status', 'Unknown')}"
                )
                logger.warning(
                    f"Last known enemy status: {self.lastInfo.get('enemy_status', 'Unknown')}"
                )
                logger.warning("\nState history before closure:")
                for i, state in enumerate(last_states):
                    logger.warning(f"  State {i+1}: {state}")
                logger.warning("==============================\n")
                
            # Update training statistics
            self.training_stats["episodes_run"] += 1
            self.training_stats["episode_rewards"].append(self.episode_reward)
            
            # Determine win/loss
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
                
            # Clean up
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
        """Ensure all required keys exist in info dictionary"""
        required_keys = {
            "continue_timer": 0,
            "round_timer": 0,
            "enemy_health": 100,
            "enemy_x_position": 200,
            "enemy_y_position": 0,
            "enemy_matches_won": 0,
            "enemy_status": Lobby.STANDING_STATUS,
            "enemy_character": 0,
            "health": 100,
            "x_position": 100,
            "y_position": 0,
            "status": Lobby.STANDING_STATUS,
            "matches_won": 0,
            "score": 0,
        }
        for key, default_value in required_keys.items():
            if key not in info:
                info[key] = default_value
        return info

    def enterFrameInputs(self):
        """Apply agent's inputs over multiple frames"""
        for frame in self.frameInputs:
            step_result = self.environment.step(frame)
            if len(step_result) == 4:
                obs, tempReward, self.done, info = step_result
            else:
                obs, tempReward, terminated, truncated, info = step_result
                self.done = terminated or truncated
                    
            # Get additional info from RAM
            info = self.read_ram_values(info)
            
            # Always accumulate reward even if done
            self.lastReward += tempReward
            
            # Check for game-ending conditions
            if info.get("health", 100) <= 0 or info.get("enemy_health", 100) <= 0:
                self.done = True
                logger.info(
                    f"Game terminated: Player health={info.get('health', 'Unknown')}, Enemy health={info.get('enemy_health', 'Unknown')}"
                )
                    
            if self.done:
                return info, obs
                    
            # Render if requested
            if self.render:
                self.environment.render()
                time.sleep(Lobby.FRAME_RATE)
                    
        return info, obs


    def executeTrainingRun(self, review=True, episodes=1):
        """Run a complete training session"""
        start_time = time.time()
        self.training_stats["session_wins"] = 0
        self.training_stats["session_losses"] = 0
        self.training_stats["session_start_time"] = time.time()

        # Training loop across episodes
        for episodeNumber in tqdm(range(episodes), desc="Training Episodes"):
            logger.info(f"\n=== Starting episode {episodeNumber+1}/{episodes} ===")
            
            # Get available states
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
                    
            # Reset episode state
            self.episode_steps = 0
            self.episode_reward = 0
            
            # play eash state opponent
            for state in states:
                # log state
                logger.info(f"Loading state: {state}")
                try:
                    # Play the state
                    self.play(state=state)
                    
                    # so 1 state then we review the fight and save
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

        # Print final stats
        if hasattr(self.players[0], "printFinalStats"):
            self.players[0].printFinalStats()

        # Print training summary
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
        
        # Calculate session statistics
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
        
        # Compare to previous sessions if available
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
                    
        # Log statistics about accumulated data
        accumulated_stats = "No"
        if hasattr(self.players[0], "total_timesteps"):
            if (
                hasattr(self.players[0], "loaded_stats")
                and self.players[0].loaded_stats
            ):
                accumulated_stats = "Yes (--resume)"
        logger.info(f"Accumulated stats: {accumulated_stats}")
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
        # Calculate training efficiency
        steps_per_second = (
            self.training_stats["total_steps"] / total_time if total_time > 0 else 0
        )
        logger.info(f"Training efficiency: {steps_per_second:.2f} steps/second")
        
        # Log agent-specific statistics
        if hasattr(self.players[0], "total_timesteps"):
            logger.info(
                f"Agent's accumulated training timesteps: {self.players[0].total_timesteps}"
            )
            
        # Analyze reward trends
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
                
            # Evaluate learning progress
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
    """Create a default state file if none exists"""
    state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "default.state")
    
    try:
        env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
        env.reset()
        
        # Let the game run for a few frames to stabilize
        for _ in range(10):
            env.step([0] * len(env.buttons))
            
        # Get the state data
        state_data = env.em.get_state()
        
        # Save as both .state and plain file (both formats needed)
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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the Street Fighter II AI training lobby with CUDA support"
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="Boolean flag for if the user wants the game environment to render during play",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=10,
        help="Integer representing the number of training rounds to go through, checkpoints are made at the end of each episode",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the instance that will be used when saving the model or its training logs",
    )
    parser.add_argument(
        "-re",
        "--resume",
        action="store_true",
        help="Boolean flag for loading a pre-existing model and stats with higher exploration for continued training",
    )
    parser.add_argument(
        "--create-state",
        action="store_true",
        help="Create a default state before running",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Disable GPU usage even if available",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show more detailed progress during training",
    )
    args = parser.parse_args()
    
    # Handle GPU disabling
    if args.disable_gpu and len(physical_devices) > 0:
        logger.info("GPU usage manually disabled")
        tf.config.set_visible_devices([], "GPU")
        
    # Create default state if requested
    if args.create_state:
        logger.info("Creating default state...")
        create_default_state()
        
    # Create agent
    agent = DeepQAgent(
        stateSize=40,
        resume=args.resume,
        name=args.name,
    )
    
    # Create and run lobby
    lobby = Lobby(render=args.render)
    lobby.addPlayer(agent)
    lobby.executeTrainingRun(episodes=args.episodes)