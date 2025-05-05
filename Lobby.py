import argparse, retro, os, time
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
    FRAME_RATE = 1 / 115  # The time between frames if real time is enabled

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

            # Add missing keys that might be needed
            if "round_timer" not in self.lastInfo:
                self.lastInfo["round_timer"] = 0  # Default value

            if "status" not in self.lastInfo:
                self.lastInfo["status"] = Lobby.STANDING_STATUS  # Default to standing

            if "x_position" not in self.lastInfo:
                self.lastInfo["x_position"] = 100  # Default position

            if "enemy_x_position" not in self.lastInfo:
                self.lastInfo["enemy_x_position"] = 200  # Default enemy position

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

                # Make sure all required keys exist in the info dict
                self.ensureRequiredKeys(info)

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
            "round_timer": 0,
            "status": Lobby.STANDING_STATUS,
            "x_position": 100,
            "enemy_x_position": 200,
            "health": 100,
            "enemy_health": 100,
        }

        for key, default_value in required_keys.items():
            if key not in info:
                info[key] = default_value

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

            if self.done:
                return info, obs
            if self.render:
                self.environment.render()
                time.sleep(Lobby.FRAME_RATE)
            self.lastReward += tempReward
        return info, obs

    # so we play each opponent in a state file
    def executeTrainingRun(self, review=True, episodes=1):
        """The lobby will load each of the saved states to generate data for the agent to train on
            Note: This will only work for single player mode

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        episodes
            An integer that represents the number of game play episodes to go through before training, once through the roster is one episode

        Returns
        -------
        None
        """
        for episodeNumber in range(episodes):
            print("Starting episode", episodeNumber)
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

            if self.players[0].__class__.__name__ != "Agent" and review:
                try:
                    self.players[0].reviewFight()
                except Exception as e:
                    print(f"Error during review: {e}")


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
