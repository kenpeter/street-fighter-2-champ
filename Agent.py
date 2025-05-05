# retro
import argparse, retro, threading, os, numpy, time, random

# dequeue
from collections import deque

# keras (like pytorch)
from tensorflow.python import keras

# keras has load model
from keras.models import load_model

# some agent move
from DefaultMoveList import Moves


class Agent:
    """Abstract class that user created Agents should inherit from.
    Contains helper functions for launching training environments and generating training data sets.
    """

    """
        [
            0: OBSERVATION,         # image/frame at time t
            1: STATE,               # internal game state at time t
            2: ACTION,              # action taken at time t
            3: REWARD,              # reward received from that action
            4: NEXT_OBSERVATION,    # image/frame at time t+1
            5: NEXT_STATE,          # internal state at time t+1
            6: DONE                 # episode completion flag
        ]

    """

    # Global constants keeping track of some input lag for some directional movements
    # Moves following these inputs will not be picked up unless input after the lag

    # The indices representing what each index in a training point represent

    # a single frame?
    OBSERVATION_INDEX = 0  # The current display image of the game state

    # agent state?
    STATE_INDEX = 1  # The state the agent was presented with

    # agent takes this action
    ACTION_INDEX = 2  # The action the agent took

    # we do have reward here
    REWARD_INDEX = 3  # The reward the agent received for that action

    # the next obs index
    NEXT_OBSERVATION_INDEX = (
        4  # The current display image of the new state the action led to
    )

    # next state index
    NEXT_STATE_INDEX = 5  # The next state that the action led to

    # done index
    DONE_INDEX = 6  # A flag signifying if the game is over

    # max frame agent can remember
    MAX_DATA_LENGTH = 50000  # Max number of decision frames the Agent can remember from a fight, average is about 2000 per fight

    DEFAULT_MODELS_DIR_PATH = "../models"  # Default path to the dir where the trained models are saved for later access
    DEFAULT_LOGS_DIR_PATH = "../logs"  # Default path to the dir where training logs are saved for user review

    ### End of static variables

    ### Object methods

    def __init__(self, load=False, name=None, moveList=Moves):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        load
            A boolean flag that specifies whether to initialize the model from scratch or load in a pretrained model

        name
            A string representing the name of the model that will be used when saving the model and the training logs
            Defaults to the class name if none are provided

        moveList
            An enum class that contains all of the allowed moves the Agent can perform

        Returns
        -------
        None
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.prepareForNextFight()
        self.moveList = moveList

        if self.__class__.__name__ != "Agent":
            self.model = (
                self.initializeNetwork()
            )  # Only invoked in child subclasses, Agent has no network
            if load:
                self.loadModel()

    # remember the fight
    def prepareForNextFight(self):
        """Clears the memory of the fighter so it can prepare to record the next fight"""
        self.memory = deque(
            maxlen=Agent.MAX_DATA_LENGTH
        )  # Double ended queue that stores states during the game

    # get random move
    # so a move contains multi frame
    def getRandomMove(self, info):
        """Returns a random set of button inputs

        Parameters
        ----------
        info
            Metadata dictionary about the current game state from the RAM

        Returns
        -------
        moveName.value
            An integer representing the move from the move list that was selected

        frameInputs
            A set of frame inputs where each number corresponds to a set of button inputs in the action space.
        """
        moveName = random.choice(
            list(self.moveList)
        )  # Take random sample of all the button press inputs the Agent could make
        frameInputs = self.convertMoveToFrameInputs(moveName, info)
        return moveName.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Converts the desired move into a series of frame inputs in order to acomplish that move

        Parameters
        ----------
        move
            enum type named after the move to be performed
            is used as the key into the move to inputs dic

        info
            Metadata dictionary about the current game state from the RAM

        Returns
        -------
        frameInputs
            An iterable frame inputs object containing the frame by frame input buffer for the move
        """
        frameInputs = self.moveList.getMoveInputs(move)
        frameInputs = self.formatInputsForDirection(move, frameInputs, info)
        return frameInputs

    def formatInputsForDirection(self, move, frameInputs, info):
        """Converts special move directional inputs to account for the player direction so they properly execute

        Parameters
        ----------
        move
            enum type named after the move to be performed
            is used as the key into the move to inputs dic

        frameInputs
            An array containing the series of frame inputs for the desired move
            In the case of a special move it has two sets of possible inputs

        info
            Information about the current game state we will pull the player
            and opponent position from

        Returns
        -------

        frameInputs
            An iterable frame inputs object containing the frame by frame input buffer for the move
        """
        if not self.moveList.isDirectionalMove(move):
            return frameInputs

        # Make sure these keys exist to avoid errors
        if "x_position" not in info:
            info["x_position"] = 100  # Default position

        if "enemy_x_position" not in info:
            info["enemy_x_position"] = 200  # Default enemy position

        if info["x_position"] < info["enemy_x_position"]:
            return (
                frameInputs[0]
                if isinstance(frameInputs, list) and len(frameInputs) > 0
                else frameInputs
            )
        else:
            return (
                frameInputs[1]
                if isinstance(frameInputs, list) and len(frameInputs) > 1
                else frameInputs
            )

        return frameInputs

    # step = [obs, state, last action, reward, next obs, next state, done]
    def recordStep(self, step):
        """Records the last observation, action, reward and the resultant observation about the environment for later training
        Parameters
        ----------
        step
            A tuple containing the following elements:

            observation
                The current display image in the form of a 2D array containing RGB values of each pixel

            state
                The state the Agent was presented with before it took an action.
                A dictionary containing tagged RAM data

            lastAction
                Integer representing the last move from the move list the Agent chose to pick

            reward
                The reward the agent received for taking that action

            nextObservation
                The resultant display image in the form of a 2D array containing RGB values of each pixel

            nextState
                The state that the chosen action led to

            done
                Whether or not the new state marks the completion of the emulation

        Returns
        -------
        None
        """
        self.memory.append(
            step
        )  # Steps are stored as tuples to avoid unintended changes

    # review a fight
    def reviewFight(self):
        """The Agent goes over the data collected from it's last fight, prepares it, and then runs through one epoch of training on the data"""
        data = self.prepareMemoryForTraining(self.memory)
        self.model = self.trainNetwork(
            data, self.model
        )  # Only invoked in child subclasses, Agent does not learn
        self.saveModel()
        self.prepareForNextFight()

    # load the model
    def loadModel(self):
        # Check if the model file exists with different extensions
        model_path = f"../models/{self.name}Model"

        # Try with different extensions
        for ext in [".keras", ".weights.h5", ".h5"]:
            full_path = model_path + ext
            if os.path.exists(full_path):
                print(f"Found model file: {full_path}")
                try:
                    self.model.load_weights(full_path)
                    print("Model loaded successfully!")
                    return
                except Exception as e:
                    print(f"Error loading model: {e}")

        # If we get here, no valid model file was found
        print(f"No valid model file found at {model_path}[.keras/.weights.h5/.h5]")
        print("Starting with a new model.")

    def saveModel(self):
        """Saves the currently trained model in the default naming convention ../models/{Instance_Name}Model
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create the models directory if it doesn't exist
        os.makedirs(Agent.DEFAULT_MODELS_DIR_PATH, exist_ok=True)

        # Fix the model filename to include the required extension
        model_path = os.path.join(
            Agent.DEFAULT_MODELS_DIR_PATH, self.getModelName() + ".weights.h5"
        )

        # Save the model with the correct file extension
        self.model.save_weights(model_path)
        print(f"Model weights saved to {model_path}")

        # Save training logs
        os.makedirs(Agent.DEFAULT_LOGS_DIR_PATH, exist_ok=True)
        with open(
            os.path.join(Agent.DEFAULT_LOGS_DIR_PATH, self.getLogsName()), "a+"
        ) as file:
            if (
                hasattr(self, "lossHistory")
                and hasattr(self.lossHistory, "losses")
                and len(self.lossHistory.losses) > 0
            ):
                file.write(
                    str(sum(self.lossHistory.losses) / len(self.lossHistory.losses))
                )
                file.write("\n")

    def getModelName(self):
        """Returns the formatted model name for the current model"""
        return self.name + "Model"

    def getLogsName(self):
        """Returns the formatted log name for the current model"""
        return self.name + "Logs"

    ### End of object methods

    ### Abstract methods for the child Agent to implement
    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network after looking at the current observation

        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values

        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost, etc.
            A full list of info can be found in data.json

        Returns
        -------
        move
            Integer representing the move that was selected from the move list

        frameInputs
            A set of frame inputs where each number corresponds to a set of button inputs in the action space.
        """
        move, frameInputs = self.getRandomMove(info)
        return move, frameInputs

    def initializeNetwork(self):
        """To be implemented in child class, should initialize or load in the Agent's neural network

        Parameters
        ----------
        None

        Returns
        -------
        model
            A newly initialized model that the Agent will use when generating moves
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    # record a fight sequence for training, so repeated watch the tape?
    def prepareMemoryForTraining(self, memory):
        """To be implemented in child class, should prepare the recorded fight sequences into training data

        Parameters
        ----------
        memory
            A 2D array where each index is a recording of a state, action, new state, and reward sequence
            See readme for more details

        Returns
        -------
        data
            The prepared training data
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    def trainNetwork(self, data, model):
        """To be implemented in child class, Runs through a training epoch reviewing the training data and returns the trained model
        Parameters
        ----------
        data
            The training data for the model

        model
            The model for the function to train

        Returns
        -------
        model
            The now trained and hopefully improved model
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    ### End of Abstract methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes agent parameters.")
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="Boolean flag for if the user wants the game environment to render during play",
    )
    args = parser.parse_args()
    # import lobby class
    from Lobby import Lobby

    # test lobby
    testLobby = Lobby(render=args.render)
    # agent
    agent = Agent()
    # add player
    testLobby.addPlayer(agent)
    # execute training
    testLobby.executeTrainingRun()
