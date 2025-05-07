import argparse, retro, threading, os, numpy, time, random, math
from collections import deque
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras.losses
from DefaultMoveList import Moves
from LossHistory import LossHistory


class Agent:
    """
    Abstract class that user created Agents should inherit from.
    Contains helper functions for launching training environments and generating training data sets.
    """

    # Global constants keeping track of some input lag for some directional movements
    # Moves following these inputs will not be picked up unless input after the lag

    # The indices representing what each index in a training point represent
    OBSERVATION_INDEX = 0  # The current display image of the game state
    STATE_INDEX = 1  # The state the agent was presented with
    ACTION_INDEX = 2  # The action the agent took
    REWARD_INDEX = 3  # The reward the agent received for that action
    NEXT_OBSERVATION_INDEX = (
        4  # The current display image of the new state the action led to
    )
    NEXT_STATE_INDEX = 5  # The next state that the action led to
    DONE_INDEX = 6  # A flag signifying if the game is over

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

    def prepareForNextFight(self):
        """Clears the memory of the fighter so it can prepare to record the next fight"""
        self.memory = deque(
            maxlen=Agent.MAX_DATA_LENGTH
        )  # Double ended queue that stores states during the game

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

    def reviewFight(self):
        """The Agent goes over the data collected from it's last fight, prepares it, and then runs through one epoch of training on the data"""
        data = self.prepareMemoryForTraining(self.memory)
        self.model = self.trainNetwork(
            data, self.model
        )  # Only invoked in child subclasses, Agent does not learn
        self.saveModel()
        self.prepareForNextFight()

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


class DeepQAgent(Agent):
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm to learn street fighter 2"""

    # so full train still 10% to explode
    EPSILON_MIN = 0.1  # Minimum exploration rate for a trained model

    # decrease slowly (become deterministic)
    DEFAULT_EPSILON_DECAY = (
        0.999  # How fast the exploration rate falls as training persists
    )

    # Q(s, a) = r + γ * max(Q(s’, a’))
    DEFAULT_DISCOUNT_RATE = (
        0.98  # How much future rewards influence the current decision of the model
    )

    # 1. big val, learn fast, unstable
    # 2. small val, learn slow, stable
    DEFAULT_LEARNING_RATE = 0.0001

    """
        stateIndices = {
            512: 0,  # Standing neutral
            514: 1,  # Crouching
            516: 2,  # Jumping
            520: 3,  # Blocking
            522: 4,  # Normal attack
            524: 5,  # Special attack
            526: 6,  # Hit stun or dizzy
            532: 7   # Being thrown
        }
    """

    # Mapping between player state values and their one hot encoding index
    stateIndices = {
        512: 0,
        514: 1,
        516: 2,
        518: 3,
        520: 4,
        522: 5,
        524: 6,
        526: 7,
        532: 8,
    }

    """
        0: Player KO (health reached zero)
        528/530: Round timer expired
        1024-1032: Various match conclusion states (win/loss animations)
    """
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]

    # action button
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]

    # can tailor for huge err and small err
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss to use as the loss function for the model"""
        import tensorflow as tf

        error = y_true - y_pred
        cond = tf.abs(error) <= clip_delta

        squared_loss = 0.5 * tf.square(error)
        quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (
            tf.abs(error) - clip_delta
        )

        return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

    def __init__(
        self,
        stateSize=32,
        load=False,
        resume=False,
        epsilon=1,
        name=None,
        moveList=Moves,
    ):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        stateSize
            The number of features that will be fed into the Agent's network

        load
            A boolean flag that specifies whether to initialize the model from scratch or load in a pretrained model

        resume
            A boolean flag that specifies whether to load a pretrained model but with higher exploration rate for continued training

        epsilon
            The initial exploration value to assume when the model is initialized. If a model is lodaed this is set
            to the minimum value

        name
            A string representing the name of the agent that will be used when saving the model and training logs
            Defaults to the class name if none is provided

        moveList
            An enum class that contains all of the allowed moves the Agent can perform

        Returns
        -------
        None
        """
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE  # discount rate

        if load and not resume:
            # Standard load mode - minimum exploration
            self.epsilon = DeepQAgent.EPSILON_MIN
        elif resume:
            # Resume mode - higher exploration for continued training
            # Set epsilon to a moderate value (0.3) to balance exploration and exploitation
            self.epsilon = 0.9
        else:
            # Fresh training mode - maximum exploration
            self.epsilon = epsilon

        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        super(DeepQAgent, self).__init__(load=load, name=name, moveList=moveList)

    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network after looking at the current observation

        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values

        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost, etc.
            A full list of info can be found in data.json
            so the info is from data.json

        Returns
        -------
        move
            An integer representing the move selected from the move list

        frameInputs
            A set of frame inputs where each number corresponds to a set of button inputs in the action space.
        """
        if numpy.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs
        else:
            stateData = self.prepareNetworkInputs(info)
            # the model predict reward
            predictedRewards = self.model.predict(stateData)[0]
            # reward become move?
            move = numpy.argmax(predictedRewards)
            frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
            return move, frameInputs

    # init the network
    def initializeNetwork(self):
        """Initializes a Neural Net for a Deep-Q learning Model

        Parameters
        ----------
        None

        Returns
        -------
        model
            The initialized neural network model that Agent will interface with to generate game moves
        """
        model = Sequential()
        model.add(Dense(48, input_dim=self.stateSize, activation="relu"))
        model.add(Dense(96, activation="relu"))
        model.add(Dense(192, activation="relu"))
        model.add(Dense(96, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.actionSize, activation="linear"))
        model.compile(
            loss=DeepQAgent._huber_loss, optimizer=Adam(learning_rate=self.learningRate)
        )

        print("Successfully initialized model")
        return model

    def prepareMemoryForTraining(self, memory):
        """prepares the recorded fight sequences into training data

        Parameters
        ----------
        memory
            A 2D array where each index is a recording of a state, action, new state, and reward sequence
            See readme for more details

        Returns
        -------
        data
            The prepared training data in whatever from the model needs to train
            DeepQ needs a state, action, and reward sequence to train on
            The observation data is thrown out for this model for training
        """
        data = []
        for step in self.memory:
            data.append(
                [
                    self.prepareNetworkInputs(step[Agent.STATE_INDEX]),
                    step[Agent.ACTION_INDEX],
                    # so we access the reward part
                    step[Agent.REWARD_INDEX],
                    step[Agent.DONE_INDEX],
                    self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX]),
                ]
            )

        return data

    def prepareNetworkInputs(self, step):
        """Generates a feature vector from the current game state information to feed into the network

        Parameters
        ----------
        step
            A given set of state information from the environment

        Returns
        -------
        feature vector
            An array extracted from the step that is the same size as the network input layer
            Takes the form of a 1 x 30 array. With the elements:
            enemy_health, enemy_x, enemy_y, 8 one hot encoded enemy state elements,
            8 one hot encoded enemy character elements, player_health, player_x, player_y, and finally
            8 one hot encoded player state elements.
        """
        feature_vector = []

        # enemy health, e-x, e-y, e-status
        feature_vector.append(step["enemy_health"])
        feature_vector.append(step["enemy_x_position"])
        feature_vector.append(step["enemy_y_position"])

        # one hot encode enemy state
        # enemy_status - 512 if standing, 514 if crouching, 516 if jumping, 518 blocking, 522 if normal attack, 524 if special attack, 526 if hit stun or dizzy, 532 if thrown
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        if step["enemy_status"] not in DeepQAgent.doneKeys:
            oneHotEnemyState[DeepQAgent.stateIndices[step["enemy_status"]]] = 1
        feature_vector += oneHotEnemyState

        # one hot encode enemy character
        oneHotEnemyChar = [0] * 8
        oneHotEnemyChar[step["enemy_character"]] = 1
        feature_vector += oneHotEnemyChar

        # Player Data
        feature_vector.append(step["health"])
        feature_vector.append(step["x_position"])
        feature_vector.append(step["y_position"])

        # player_status - 512 if standing, 514 if crouching, 516 if jumping, 520 blocking, 522 if normal attack, 524 if special attack, 526 if hit stun or dizzy, 532 if thrown
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        if step["status"] not in DeepQAgent.doneKeys:
            oneHotPlayerState[DeepQAgent.stateIndices[step["status"]]] = 1
        feature_vector += oneHotPlayerState

        feature_vector = numpy.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def trainNetwork(self, data, model):
        # Limit the batch size to prevent too many iterations
        max_samples = min(
            len(data), 100
        )  # Process at most 100 samples per training run
        minibatch = random.sample(data, max_samples)

        self.lossHistory.losses_clear()
        batch_count = 0
        max_batches = 20  # Set a maximum number of batches

        for state, action, reward, done, next_state in minibatch:
            if batch_count >= max_batches:
                break

            modelOutput = model.predict(state)[0]
            if not done:
                reward = reward + self.gamma * numpy.amax(model.predict(next_state)[0])

            modelOutput[action] = reward
            modelOutput = numpy.reshape(modelOutput, [1, self.actionSize])
            model.fit(
                state, modelOutput, epochs=1, verbose=0, callbacks=[self.lossHistory]
            )
            batch_count += 1

        if self.epsilon > DeepQAgent.EPSILON_MIN:
            self.epsilon *= self.epsilonDecay
        return model


from keras.utils import get_custom_objects

loss = DeepQAgent._huber_loss
get_custom_objects().update({"_huber_loss": loss})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes agent parameters.")
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
        help="Name of the instance that will be used when saving the model or it's training logs",
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
    args = parser.parse_args()
    if args.load:
        qAgent = DeepQAgent(load=True, name=args.name)
    elif args.resume:
        qAgent = DeepQAgent(load=True, resume=True, name=args.name)
    else:
        qAgent = DeepQAgent(load=False, name=args.name)
    from Lobby import Lobby

    testLobby = Lobby(render=args.render)
    testLobby.addPlayer(qAgent)
    testLobby.executeTrainingRun(
        episodes=args.episodes, background_training=args.background
    )
