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

    @staticmethod
    def load_state_file(state_name):
        """Helper method to load a state file from the correct location"""
        # Base directory for state files
        directory = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")

        # Try different possible paths
        paths_to_try = [
            os.path.join(directory, f"{state_name}.state"),  # with .state extension
            os.path.join(directory, state_name),  # without extension
            state_name,  # just the name (default states)
        ]

        for path in paths_to_try:
            if os.path.exists(path):
                print(f"Found state file at: {path}")
                with open(path, "rb") as f:
                    state_data = f.read()
                return path, state_data

        # If we get here, we couldn't find the state file
        print(f"Warning: Could not find state file for '{state_name}'")
        return None, None

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
                try:
                    # Try to create a default state
                    create_default_state(directory)
                    return ["default"]
                except:
                    print("Could not create a default state")

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

        # Register our state directory with retro
        state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
        try:
            retro.data.add_custom_path(state_dir)
            print(f"Registered state directory: {state_dir}")
        except Exception as e:
            print(f"Warning: Could not register state directory: {e}")

    def initEnvironment(self, state):
        print(f"Initializing environment with state: {state}")
        try:
            # First try to use the built-in state with just the name
            try:
                self.environment = retro.make(
                    game=self.game, state=state, players=self.mode.value
                )
                print(f"Successfully created environment with built-in state: {state}")
            except Exception as state_error:
                print(f"Could not use built-in state '{state}': {state_error}")

                # Try with the file path
                state_path = os.path.join(
                    os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
                    f"{state}.state",
                )

                if os.path.exists(state_path):
                    print(f"Using state file at: {state_path}")
                    # Create environment without state first
                    self.environment = retro.make(
                        game=self.game, players=self.mode.value
                    )

                    # Reset the environment
                    self.environment.reset()

                    # Load the state manually
                    with open(state_path, "rb") as f:
                        state_data = f.read()

                    self.environment.em.set_state(state_data)
                    print(f"Manually loaded state from: {state_path}")
                else:
                    # If we still can't find the state, create a default environment
                    print(f"State file not found at: {state_path}")
                    print(f"Creating default environment without state")
                    self.environment = retro.make(
                        game=self.game, players=self.mode.value
                    )

            # Apply the discretizer
            self.environment = StreetFighter2Discretizer(self.environment)
            print("Applied StreetFighter2Discretizer")

            # Reset the environment if needed
            obs = self.environment.reset()
            print("Environment reset")

            # Take a step
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

            # Check if expected keys are missing
            missing_keys = []
            expected_keys = ["round_timer", "status", "health", "enemy_health"]
            for key in expected_keys:
                if key not in self.lastInfo:
                    missing_keys.append(key)
            if missing_keys:
                print(f"WARNING: Expected keys missing from info: {missing_keys}")

            self.lastAction, self.frameInputs = 0, [Lobby.NO_ACTION]
            self.currentJumpFrame = 0

            # Wait for an actionable state
            counter = 0
            while (
                not self.isActionableState(self.lastInfo, Lobby.NO_ACTION)
                and counter < 100
            ):
                # Handle both old and new retro API return patterns
                step_result = self.environment.step(Lobby.NO_ACTION)
                if len(step_result) == 4:  # Old pattern
                    self.lastObservation, reward, done, self.lastInfo = step_result
                    self.done = done
                else:  # New pattern with 5 returns
                    (
                        self.lastObservation,
                        reward,
                        terminated,
                        truncated,
                        self.lastInfo,
                    ) = step_result
                    self.done = terminated or truncated

                counter += 1
                if counter >= 100:
                    print("Warning: Could not reach actionable state after 100 steps")

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
        # Check if the required keys exist in info
        if "round_timer" not in info or "status" not in info:
            print(f"WARNING: Missing keys in info: {info.keys()}")
            return True  # Return true to avoid infinite loop

        action = self.environment.get_action_meaning(action)

        # if there is a timer for the game count down
        # if not yet start the game, prevent player to play
        if info["round_timer"] == Lobby.ROUND_TIMER_NOT_STARTED:
            return False
        elif (
            # this agent status
            info["status"] == Lobby.JUMPING_STATUS
            and self.currentJumpFrame <= Lobby.JUMP_LAG
        ):
            self.currentJumpFrame += 1
            return False
        elif info["status"] == Lobby.JUMPING_STATUS and any(
            [button in action for button in Lobby.ACTION_BUTTONS]
        ):  # Have to manually track if we are in a jumping attack
            return False
        elif (
            info["status"] not in Lobby.ACTIONABLE_STATUSES
        ):  # Standing, Crouching, or Jumping
            return False
        else:
            if info["status"] != Lobby.JUMPING_STATUS and self.currentJumpFrame > 0:
                self.currentJumpFrame = 0
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
        self.initEnvironment(state)
        max_steps = 5000  # Limit the number of steps to avoid infinite loops
        step_count = 0

        while not self.done and step_count < max_steps:
            step_count += 1

            # action is an iterable object that contains an input buffer representing frame by frame inputs
            # the lobby will run through these inputs and enter each one on the appropriate frames

            # last action, frame input
            # from player(obs, lastinfo)
            self.lastAction, self.frameInputs = self.players[0].getMove(
                self.lastObservation, self.lastInfo
            )

            # Fully execute frame object and then wait for next actionable state
            self.lastReward = 0
            info, obs = self.enterFrameInputs()
            info, obs = self.waitForNextActionableState(info, obs)

            # in lobby, we record player's step
            #
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
            self.lastObservation, self.lastInfo = [
                obs,
                info,
            ]  # Overwrite after recording step so Agent remembers the previous state that led to this one

        if step_count >= max_steps:
            print(f"Warning: Reached maximum steps ({max_steps}) for state {state}")

        self.environment.close()
        if self.render:
            self.environment.viewer.close()

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

    # wait for the next actionable state
    def waitForNextActionableState(self, info, obs):
        """Wait for the next game state where the Agent can make an action

        Parameters
        ----------
        info
            The ram info received from the emulator of the last frame of the game

        obs
            The image buffer received from the emulator starting the frame after that last set of inputs

        Returns
        -------
        info
            The ram info received from the emulator after finally getting to an actionable state

        obs
            The image buffer data received from the emulator after finally getting to an actionable state

        """
        # Limit the number of steps to avoid infinite loops
        max_wait_steps = 100
        wait_step_count = 0

        # to prevent blindly action something, which is not actionable.
        while (
            not self.isActionableState(info, action=self.frameInputs[-1])
            and wait_step_count < max_wait_steps
        ):
            wait_step_count += 1

            # Handle both the old (4-return) and new (5-return) patterns
            step_result = self.environment.step(Lobby.NO_ACTION)
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

        if wait_step_count >= max_wait_steps:
            print(
                f"Warning: Could not reach actionable state after {max_wait_steps} steps"
            )

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
                    continue

            if self.players[0].__class__.__name__ != "Agent" and review == True:
                try:
                    self.players[0].reviewFight()
                except Exception as e:
                    print(f"Error during review: {e}")


def create_default_state():
    """Create a default state file"""
    state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    os.makedirs(state_dir, exist_ok=True)

    state_path = os.path.join(state_dir, "default.state")

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
