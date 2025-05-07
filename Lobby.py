import argparse, retro, os, time
import random
from enum import Enum
import sys
import tensorflow as tf
from Discretizer import StreetFighter2Discretizer
from myagent import DeepQAgent

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    try:
        # Allow TensorFlow to allocate only as much GPU memory as needed
        print(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled.")

        # Set the visible device to ensure TensorFlow uses GPU
        tf.config.set_visible_devices(physical_devices[0], "GPU")
        print(f"Set visible GPU device: {physical_devices[0].name}")

    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Will use CPU instead.")


# too many player in lobby
class Lobby_Full_Exception(Exception):
    pass


# determines how many players the lobby will request moves from before updating the game state
# 1 vs ai, or 1 vs 1
class Lobby_Modes(Enum):
    SINGLE_PLAYER = 1
    TWO_PLAYER = 2


# this is the lobby buttons
class Lobby:
    """A class that handles all of the necessary book keeping for running the gym environment.
    A number of players are added and a game state is selected and the lobby will handle
    piping in the player moves and keeping track of some relevant game information.
    """

    ### Static Variables

    # Variables relating to monitoring state and contorls
    # no action
    NO_ACTION = 0
    # movement button
    MOVEMENT_BUTTONS = ["LEFT", "RIGHT", "DOWN", "UP"]
    # x, y, z, a, b, c
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]
    # 39208 is ram address
    ROUND_TIMER_NOT_STARTED = 39208

    # agent status address 16744450 has following value
    STANDING_STATUS = 512
    CROUCHING_STATUS = 514
    JUMPING_STATUS = 516

    # because this list can do new attack follow up
    # e.g. speical move afterward, it is locked.
    ACTIONABLE_STATUSES = [STANDING_STATUS, CROUCHING_STATUS, JUMPING_STATUS]

    # Variables keeping track of the delay between these movement inputs and
    # when the next button inputs are picked ups
    # one action, then another action gap
    JUMP_LAG = 4

    # time between 2 frame
    FRAME_RATE = 1 / 300  # Slow enough for human viewing

    ### End of static variables

    ### Static Methods

    def getStates():
        """Static method that gets and returns a list of all the save state names that can be loaded"""
        directory = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            return []

        try:
            files = os.listdir(directory)
            states = [
                os.path.splitext(file)[0] for file in files if file.endswith(".state")
            ]
            if not states:
                return []
            print(f"Found states: {states}")
            return states
        except Exception as e:
            print(f"Error getting states: {e}")
            return []

    ### End of static methods

    def __init__(
        self,
        game="StreetFighterIISpecialChampionEdition-Genesis",
        render=False,
        mode=Lobby_Modes.SINGLE_PLAYER,
    ):
        """Initializes the agent and the underlying neural network"""
        self.game = game
        self.render = render
        self.mode = mode
        self.clearLobby()
        self.environment = None

        # Initialize training statistics
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
        """Read RAM values and populate the info dictionary"""
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
                    print(f"Error reading RAM for {key}: {e}")

        except Exception as e:
            print(f"Error reading RAM: {e}")

        return self.ensureRequiredKeys(info)

    def initEnvironment(self, state):
        print(f"Initializing environment with state: {state}")
        try:
            if self.environment is not None:
                try:
                    self.environment.close()
                    print("Closed existing environment")
                except Exception as close_error:
                    print(f"Warning when closing environment: {close_error}")
                self.environment = None

            self.environment = retro.make(game=self.game, players=self.mode.value)
            self.environment.reset()

            state_path = os.path.join(
                os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
                f"{state}.state",
            )

            if os.path.exists(state_path):
                print(f"Loading state from: {state_path}")
                try:
                    with open(state_path, "rb") as f:
                        state_data = f.read()
                    self.environment.em.set_state(state_data)
                    print(f"Loaded state successfully")
                except Exception as state_error:
                    print(f"Warning when loading state: {state_error}")

            self.environment = StreetFighter2Discretizer(self.environment)
            print("Applied StreetFighter2Discretizer")

            step_result = self.environment.step(Lobby.NO_ACTION)
            if len(step_result) == 4:
                self.lastObservation, reward, done, self.lastInfo = step_result
                self.done = done
            else:
                self.lastObservation, reward, terminated, truncated, self.lastInfo = (
                    step_result
                )
                self.done = terminated or truncated

            print("Environment stepped with NO_ACTION")
            print(f"Info keys available: {list(self.lastInfo.keys())}")

            self.lastInfo = self.read_ram_values(self.lastInfo)

            print(f"Info keys after RAM reading: {list(self.lastInfo.keys())}")

            self.lastAction, self.frameInputs = 0, [Lobby.NO_ACTION]
            self.currentJumpFrame = 0

            # Initialize episode-specific metrics
            self.episode_steps = 0
            self.episode_reward = 0
            self.initial_health = self.lastInfo.get("health", 100)
            self.initial_enemy_health = self.lastInfo.get("enemy_health", 100)

        except Exception as e:
            print(f"Error initializing environment: {e}")
            raise

    def addPlayer(self, newPlayer):
        """Adds a new player to the player list of active players in this lobby"""
        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return
        raise Lobby_Full_Exception(
            "Lobby has already reached the maximum number of players"
        )

    def clearLobby(self):
        """Clears the players currently inside the lobby's play queue"""
        self.players = [None] * self.mode.value

    def isActionableState(self, info, action=0):
        """Determines if the Agent has control over the game in it's current state"""
        return True

    def play(self, state):
        """The Agent will load the specified save state and play through it until finished"""
        try:
            self.initEnvironment(state)
            max_steps = 2000
            step_count = 0

            while not self.done and step_count < max_steps:
                step_count += 1
                self.episode_steps += 1
                self.training_stats["total_steps"] += 1

                # Wrap prediction in GPU context if GPU is available
                if len(physical_devices) > 0:
                    with tf.device("/GPU:0"):
                        self.lastAction, self.frameInputs = self.players[0].getMove(
                            self.lastObservation, self.lastInfo
                        )
                else:
                    self.lastAction, self.frameInputs = self.players[0].getMove(
                        self.lastObservation, self.lastInfo
                    )

                self.lastReward = 0
                info, obs = self.enterFrameInputs()

                # Track episode reward
                self.episode_reward += self.lastReward

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
                self.lastObservation, self.lastInfo = [obs, info]

                if step_count % 100 == 0:
                    print(
                        f"Step {step_count}, Player health: {info['health']}, Enemy health: {info['enemy_health']}"
                    )

            # Update episode statistics
            self.training_stats["episodes_run"] += 1
            self.training_stats["episode_rewards"].append(self.episode_reward)

            # Determine win/loss by comparing health
            if self.lastInfo.get("health", 0) > self.lastInfo.get("enemy_health", 0):
                self.training_stats["wins"] += 1
                self.training_stats["session_wins"] += 1
                print("Episode result: WIN")
            else:
                self.training_stats["losses"] += 1
                self.training_stats["session_losses"] += 1
                print("Episode result: LOSS")

            print(f"Episode steps: {self.episode_steps}")
            print(f"Episode reward: {self.episode_reward}")
            print(f"Total steps so far: {self.training_stats['total_steps']}")

            if step_count >= max_steps:
                print(
                    "WARNING: Episode terminated due to step limit, not game completion"
                )
            else:
                print("Episode completed naturally")

            if self.environment is not None:
                try:
                    self.environment.close()
                    self.environment = None
                except:
                    pass

        except Exception as e:
            print(f"Error playing state {state}: {e}")
            if self.environment is not None:
                try:
                    self.environment.close()
                    self.environment = None
                except:
                    pass

    def ensureRequiredKeys(self, info):
        """Make sure all required keys exist in the info dictionary"""
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
        """Enter each of the frame inputs in the input buffer inside the last action object supplied by the Agent"""
        for frame in self.frameInputs:
            step_result = self.environment.step(frame)
            if len(step_result) == 4:
                obs, tempReward, self.done, info = step_result
            else:
                obs, tempReward, terminated, truncated, info = step_result
                self.done = terminated or truncated

            info = self.read_ram_values(info)

            if self.done:
                return info, obs
            if self.render:
                self.environment.render()
                time.sleep(Lobby.FRAME_RATE)
            self.lastReward += tempReward
        return info, obs

    def executeTrainingRun(self, review=True, episodes=1, background_training=True):
        """The lobby will load each of the saved states to generate data for the agent to train on"""
        start_time = time.time()

        def training_thread_function(agent):
            try:
                print("Starting training in background thread...")
                # Ensure background training uses GPU if available
                if len(physical_devices) > 0:
                    print("Using GPU for background training")
                    with tf.device("/GPU:0"):
                        agent.reviewFight()
                else:
                    agent.reviewFight()
                print("Background training completed!")
            except Exception as e:
                print(f"Error during background training: {e}")
                import traceback

                traceback.print_exc()

        if background_training and episodes > 1:
            display_episode = random.randint(0, episodes - 1)
            print(f"Will render episode {display_episode} while training in background")
        else:
            display_episode = 0

        original_render = self.render

        for episodeNumber in range(episodes):
            print(f"\n=== Starting episode {episodeNumber+1}/{episodes} ===")

            if background_training:
                self.render = episodeNumber == display_episode

            states = Lobby.getStates()

            if not states:
                print("No state files found. Creating a default state...")
                try:
                    create_default_state()
                    states = ["default"]
                except Exception as e:
                    print(f"Error creating default state: {e}")
                    print(
                        "Please create at least one state file before running training."
                    )
                    return

            # Reset episode-specific metrics
            self.episode_steps = 0
            self.episode_reward = 0

            for state in states:
                print(f"Loading state: {state}")
                try:
                    self.play(state=state)
                except Exception as e:
                    print(f"Error playing state {state}: {e}")
                    if self.environment is not None:
                        try:
                            self.environment.close()
                            self.environment = None
                        except:
                            pass
                    continue

            if (
                episodeNumber == display_episode
                and background_training
                and review
                and self.players[0].__class__.__name__ != "Agent"
            ):
                import threading

                training_thread = threading.Thread(
                    target=training_thread_function, args=(self.players[0],)
                )
                training_thread.daemon = True
                training_thread.start()

        self.render = original_render

        if (
            not background_training
            and review
            and self.players[0].__class__.__name__ != "Agent"
        ):
            try:
                print("Starting review process...")
                # Use GPU for training if available
                if len(physical_devices) > 0:
                    print("Using GPU for review...")
                    with tf.device("/GPU:0"):
                        self.players[0].reviewFight()
                else:
                    self.players[0].reviewFight()
                print("Review completed")
            except Exception as e:
                print(f"Error during review: {e}")
                import traceback

                traceback.print_exc()

        # Print final training summary
        if hasattr(self.players[0], "printFinalStats"):
            self.players[0].printFinalStats()

        # Print lobby training statistics
        total_time = time.time() - start_time
        win_rate = (
            (self.training_stats["wins"] / self.training_stats["episodes_run"]) * 100
            if self.training_stats["episodes_run"] > 0
            else 0
        )

        print("\n========= TRAINING SESSION SUMMARY =========")
        print(f"Total training steps: {self.training_stats['total_steps']}")
        print(f"Total episodes: {self.training_stats['episodes_run']}")

        # Overall win/loss record
        print(
            f"Total Win/Loss Record: {self.training_stats['wins']}W - {self.training_stats['losses']}L ({win_rate:.2f}%)"
        )

        # Current session win/loss record
        session_win_rate = (
            (self.training_stats["session_wins"] / episodes) * 100
            if episodes > 0
            else 0
        )
        print(
            f"Current session record: {self.training_stats['session_wins']}W - {self.training_stats['session_losses']}L ({session_win_rate:.2f}%)"
        )

        # Track progress of win rate over time
        if hasattr(self.players[0], "loaded_stats") and self.players[0].loaded_stats:
            previous_wins = (
                self.training_stats["wins"] - self.training_stats["session_wins"]
            )
            previous_losses = (
                self.training_stats["losses"] - self.training_stats["session_losses"]
            )
            previous_episodes = self.training_stats["episodes_run"] - episodes

            if previous_episodes > 0:
                previous_win_rate = (previous_wins / previous_episodes) * 100
                win_rate_change = session_win_rate - previous_win_rate
                print(
                    f"Win rate change: {win_rate_change:+.2f}% (Previous: {previous_win_rate:.2f}%)"
                )

                if win_rate_change > 5:
                    print("Performance trend: STRONG IMPROVEMENT")
                elif win_rate_change > 0:
                    print("Performance trend: SLIGHT IMPROVEMENT")
                elif win_rate_change > -5:
                    print("Performance trend: STABLE")
                else:
                    print("Performance trend: DECLINING")

        # Track if we used resume flag to accumulate stats
        accumulated_stats = "No"
        if hasattr(self.players[0], "total_timesteps"):
            # Check if we're accumulating (resumed) or starting fresh
            if (
                hasattr(self.players[0], "loaded_stats")
                and self.players[0].loaded_stats
            ):
                accumulated_stats = "Yes (--resume)"

        print(f"Accumulated stats: {accumulated_stats}")
        print(f"Total training time: {total_time:.2f} seconds")

        # Calculate steps per second
        steps_per_second = (
            self.training_stats["total_steps"] / total_time if total_time > 0 else 0
        )
        print(f"Training efficiency: {steps_per_second:.2f} steps/second")

        if hasattr(self.players[0], "total_timesteps"):
            print(
                f"Agent's accumulated training timesteps: {self.players[0].total_timesteps}"
            )

        # Calculate average reward trend
        if len(self.training_stats["episode_rewards"]) >= 2:
            first_rewards = sum(
                self.training_stats["episode_rewards"][
                    : min(3, len(self.training_stats["episode_rewards"]))
                ]
            ) / min(3, len(self.training_stats["episode_rewards"]))
            last_rewards = sum(
                self.training_stats["episode_rewards"][
                    -min(3, len(self.training_stats["episode_rewards"])) :
                ]
            ) / min(3, len(self.training_stats["episode_rewards"]))
            reward_trend = last_rewards - first_rewards
            print(f"Reward trend: {reward_trend:+.2f}")

            if reward_trend > 0:
                print("Learning assessment: POSITIVE - Agent is improving")
            elif reward_trend > -1:
                print("Learning assessment: NEUTRAL - Agent performance is stable")
            else:
                print(
                    "Learning assessment: NEGATIVE - Agent may be stuck in suboptimal policy"
                )

        print("===========================================")


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

        print(f"Created default state at {state_path}")
        env.close()
        return "default"
    except Exception as e:
        print(f"Error creating default state: {e}")
        return None


if __name__ == "__main__":
    # Print CUDA information before starting
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

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
        "-l",
        "--load",
        action="store_true",
        help="Boolean flag for if the user wants to load pre-existing weights",
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
        "-b",
        "--background",
        action="store_true",
        help="Train in background while rendering a random episode",
    )
    parser.add_argument(
        "-re",
        "--resume",
        action="store_true",
        help="Boolean flag for loading a pre-existing model but with higher exploration for continued training",
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

    # Handle GPU disable option
    if args.disable_gpu and len(physical_devices) > 0:
        print("GPU usage manually disabled")
        tf.config.set_visible_devices([], "GPU")

    if args.create_state:
        print("Creating default state...")
        create_default_state()

    if args.load and not args.resume:
        agent = DeepQAgent(load=True, name=args.name)
    elif args.resume:
        agent = DeepQAgent(load=True, resume=True, name=args.name)
    else:
        agent = DeepQAgent(load=False, name=args.name)

    lobby = Lobby(render=args.render)
    lobby.addPlayer(agent)
    lobby.executeTrainingRun(
        episodes=args.episodes, background_training=args.background
    )
