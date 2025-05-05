import argparse, retro, os, time
import random
from enum import Enum
import sys
from Discretizer import StreetFighter2Discretizer


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
    # FRAME_RATE = 1 / 115  # The time between frames if real time is enabled
    FRAME_RATE = 1 / 90  # Slow enough for human viewing

    ### End of static variables

    ### Static Methods

    # need to get state
    def getStates():
        """Static method that gets and returns a list of all the save state names that can be loaded

        Parameters
        ----------
        None

        ReturnsStreetFighter2Discretizer
        -------
        states
            A list of strings where each string is the name of a different save state
        """

        directory = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")

        # Make sure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            return []

        try:
            files = os.listdir(directory)
            # Return just the state names without extension
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
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        game
            A String of the game the lobby will be making an environment of, defaults to StreetFighterIISpecialChampionEdition-Genesis

        render
            A boolean flag that specifies whether or not to visually render the game while a match is being played

        mode
            An enum type that describes whether this lobby is for single player or two player matches

        Returns
        -------
        None
        """
        self.game = game
        self.render = render
        self.mode = mode
        self.clearLobby()
        self.environment = None

        # Define memory addresses directly in the class
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
            # Get raw RAM from the unwrapped environment
            if hasattr(self.environment.unwrapped, "get_ram"):
                ram = self.environment.unwrapped.get_ram()
            elif hasattr(self.environment.unwrapped, "em") and hasattr(
                self.environment.unwrapped.em, "get_ram"
            ):
                ram = self.environment.unwrapped.em.get_ram()
            else:
                # Can't access RAM, use default values
                return self.ensureRequiredKeys(info)

            for key, address_info in self.ram_info.items():
                addr = address_info["address"]
                data_type = address_info["type"]

                # Check if address is in valid range
                if addr >= len(ram):
                    continue

                try:
                    # Extract value based on type
                    if data_type == "|u1":
                        # Unsigned 1-byte
                        value = ram[addr]
                    elif data_type == ">u2":
                        # Big-endian 2-byte unsigned int
                        if addr + 1 < len(ram):
                            value = (ram[addr] << 8) | ram[addr + 1]
                        else:
                            continue
                    elif data_type == ">i2":
                        # Big-endian 2-byte signed int
                        if addr + 1 < len(ram):
                            value = (ram[addr] << 8) | ram[addr + 1]
                            # Convert to signed if necessary
                            if value >= 32768:
                                value -= 65536
                        else:
                            continue
                    elif data_type == ">u4":
                        # Big-endian 4-byte unsigned int
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
                        # Big-endian 4-byte decimal (assuming this is a float)
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
                        # Default case
                        value = ram[addr]

                    # Store in info dict
                    info[key] = value
                except Exception as e:
                    print(f"Error reading RAM for {key}: {e}")

        except Exception as e:
            print(f"Error reading RAM: {e}")

        # Make sure all required keys exist
        return self.ensureRequiredKeys(info)

    def initEnvironment(self, state):
        print(f"Initializing environment with state: {state}")
        try:
            # Close any existing environment to prevent multiple instances error
            if self.environment is not None:
                try:
                    self.environment.close()
                    print("Closed existing environment")
                except Exception as close_error:
                    print(f"Warning when closing environment: {close_error}")
                self.environment = None

            # Create environment without state first
            self.environment = retro.make(game=self.game, players=self.mode.value)

            # Reset the environment
            self.environment.reset()

            # Try to load the state from file
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

            # Apply the discretizer
            self.environment = StreetFighter2Discretizer(self.environment)
            print("Applied StreetFighter2Discretizer")

            # Take a step to get initial observation and info
            step_result = self.environment.step(Lobby.NO_ACTION)
            if len(step_result) == 4:  # Old pattern
                self.lastObservation, reward, done, self.lastInfo = step_result
                self.done = done
            else:  # New pattern with 5 returns
                self.lastObservation, reward, terminated, truncated, self.lastInfo = (
                    step_result
                )
                self.done = terminated or truncated

            print("Environment stepped with NO_ACTION")
            print(f"Info keys available: {list(self.lastInfo.keys())}")

            # Read RAM values to populate info dictionary
            self.lastInfo = self.read_ram_values(self.lastInfo)

            print(f"Info keys after RAM reading: {list(self.lastInfo.keys())}")

            self.lastAction, self.frameInputs = 0, [Lobby.NO_ACTION]
            self.currentJumpFrame = 0

        except Exception as e:
            print(f"Error initializing environment: {e}")
            raise

    def addPlayer(self, newPlayer):
        """Adds a new player to the player list of active players in this lobby
           will throw a Lobby_Full_Exception if the lobby is full

        Parameters
        ----------
        newPlayer
            An agent object that will be added to the lobby and moves will be requested from when the lobby starts

        Returns
        -------
        None
        """
        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return

        raise Lobby_Full_Exception(
            "Lobby has already reached the maximum number of players"
        )

    # clear players currently inside the lobby queue
    def clearLobby(self):
        """Clears the players currently inside the lobby's play queue

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.players = [None] * self.mode.value

    def isActionableState(self, info, action=0):
        """Determines if the Agent has control over the game in it's current state(the Agent is in hit stun, ending lag, etc.)

        Parameters
        ----------
        info
            The RAM info of the current game state the Agent is presented with as a dictionary of keyworded values from Data.json

        action
            The last action taken by the Agent

        Returns
        -------
        isActionable
            A boolean variable describing whether the Agent has control over the given state of the game
        """
        # Always return True to avoid problems with missing keys
        return True

    def play(self, state):
        """The Agent will load the specified save state and play through it until finished, recording the fight for training

        Parameters
        ----------
        state
            A string of the name of the save state the Agent will be playing

        Returns
        -------
        None
        """
        try:
            self.initEnvironment(state)
            max_steps = 500  # Limit the number of steps
            step_count = 0

            while not self.done and step_count < max_steps:
                step_count += 1

                # Get move from the agent
                self.lastAction, self.frameInputs = self.players[0].getMove(
                    self.lastObservation, self.lastInfo
                )

                # Execute the move
                self.lastReward = 0
                info, obs = self.enterFrameInputs()

                # Record step
                self.players[0].recordStep(
                    (
                        # last obs, last info, last action, last reward, obs, info, done
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

            # Close the environment
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
        """Enter each of the frame inputs in the input buffer inside the last action object supplied by the Agent

        Parameters
        ----------
        None

        Returns
        -------
        info
            The ram information received from the emulator after the last frame input has been entered

        obs
            The image buffer data received from the emulator after entering all input frames
        """
        for frame in self.frameInputs:
            # Handle both the old (4-return) and new (5-return) patterns
            step_result = self.environment.step(frame)
            if len(step_result) == 4:  # old pattern
                obs, tempReward, self.done, info = step_result
            else:  # new pattern with 5 returns
                obs, tempReward, terminated, truncated, info = step_result
                self.done = terminated or truncated

            # Read RAM values to populate info dictionary
            info = self.read_ram_values(info)

            if self.done:
                return info, obs
            if self.render:
                self.environment.render()
                time.sleep(Lobby.FRAME_RATE)
            self.lastReward += tempReward
        return info, obs

    # so we play each opponent in a state file
    def executeTrainingRun(self, review=True, episodes=1, background_training=True):
        """The lobby will load each of the saved states to generate data for the agent to train on
            Note: This will only work for single player mode

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        episodes
            An integer that represents the number of game play episodes to go through before training, once through the roster is one episode

        background_training
            If True, will train in the background while continuing to render gameplay

        Returns
        -------
        None
        """

        def training_thread_function(agent):
            try:
                print("Starting training in background thread...")
                agent.reviewFight()
                print("Background training completed!")
            except Exception as e:
                print(f"Error during background training: {e}")
                import traceback

                traceback.print_exc()

        # Choose one random episode to display
        if background_training and episodes > 1:
            display_episode = random.randint(0, episodes - 1)
            print(f"Will render episode {display_episode} while training in background")
        else:
            display_episode = 0

        # Save original render setting to restore after training specific episodes
        original_render = self.render

        for episodeNumber in range(episodes):
            print("Starting episode", episodeNumber)

            # Only render the selected episode
            if background_training:
                self.render = episodeNumber == display_episode

            # Get available states
            states = Lobby.getStates()

            if not states:
                print("No state files found. Creating a default state...")
                try:
                    # Try to create a default state
                    create_default_state()
                    states = ["default"]
                except Exception as e:
                    print(f"Error creating default state: {e}")
                    print(
                        "Please create at least one state file before running training."
                    )
                    return

            for state in states:
                print(f"Loading state: {state}")
                try:
                    # Play using the state name
                    self.play(state=state)
                except Exception as e:
                    print(f"Error playing state {state}: {e}")
                    # Make sure environment is closed
                    if self.environment is not None:
                        try:
                            self.environment.close()
                            self.environment = None
                        except:
                            pass
                    continue

            # After the first episode, if it's the one we're rendering, start background training
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
                training_thread.daemon = (
                    True  # Make thread a daemon so it exits when main program exits
                )
                training_thread.start()

        # Restore original render setting
        self.render = original_render

        # If not training in background, do it in the main thread
        if (
            not background_training
            and review
            and self.players[0].__class__.__name__ != "Agent"
        ):
            try:
                print("Starting review process...")
                self.players[0].reviewFight()
                print("Review completed")
            except Exception as e:
                print(f"Error during review: {e}")
                import traceback

                traceback.print_exc()


def create_default_state():
    """Create a default state file"""
    state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    os.makedirs(state_dir, exist_ok=True)

    state_path = os.path.join(state_dir, "default.state")

    try:
        # Create the environment and get its state
        env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
        env.reset()

        # Take a few steps to stabilize
        for _ in range(10):
            env.step([0] * len(env.buttons))

        # Get and save the state
        state_data = env.em.get_state()
        with open(state_path, "wb") as f:
            f.write(state_data)

        # Also save without extension for compatibility
        with open(os.path.join(state_dir, "default"), "wb") as f:
            f.write(state_data)

        print(f"Created default state at {state_path}")
        env.close()
        return "default"
    except Exception as e:
        print(f"Error creating default state: {e}")
        return None


# Makes an example lobby and has a random agent play through an example training run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Street Fighter II AI training lobby"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game visually"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to train"
    )
    parser.add_argument(
        "--create-state",
        action="store_true",
        help="Create a default state before running",
    )
    args = parser.parse_args()

    if args.create_state:
        print("Creating default state...")
        create_default_state()

    testLobby = Lobby(render=args.render)
    from Agent import Agent

    agent = Agent()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(episodes=args.episodes)
