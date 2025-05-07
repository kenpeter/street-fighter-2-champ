import os
import numpy as np
import random
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

# Enable GPU memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Allow TensorFlow to allocate only as much GPU memory as needed
        print(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print(f"Error configuring GPU: {e}")


class Agent:
    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    MAX_DATA_LENGTH = 50000
    DEFAULT_MODELS_DIR_PATH = "../models"
    DEFAULT_LOGS_DIR_PATH = "../logs"

    def __init__(self, load=False, name=None, moveList=Moves):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.prepareForNextFight()
        self.moveList = moveList
        if self.__class__.__name__ != "Agent":
            self.model = self.initializeNetwork()
            if load:
                self.loadModel()

    def prepareForNextFight(self):
        self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)

    def getRandomMove(self, info):
        moveName = random.choice(list(self.moveList))
        frameInputs = self.convertMoveToFrameInputs(moveName, info)
        return moveName.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        frameInputs = self.moveList.getMoveInputs(move)
        frameInputs = self.formatInputsForDirection(move, frameInputs, info)
        return frameInputs

    def formatInputsForDirection(self, move, frameInputs, info):
        if not self.moveList.isDirectionalMove(move):
            return frameInputs
        if "x_position" not in info:
            info["x_position"] = 100
        if "enemy_x_position" not in info:
            info["enemy_x_position"] = 200
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

    def recordStep(self, step):
        self.memory.append(step)

    def reviewFight(self):
        data = self.prepareMemoryForTraining(self.memory)
        self.model = self.trainNetwork(data, self.model)
        self.saveModel()
        self.prepareForNextFight()

    def loadModel(self):
        model_path = f"../models/{self.name}Model"
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
        print(f"No valid model file found at {model_path}[.keras/.weights.h5/.h5]")
        print("Starting with a new model.")

    def saveModel(self):
        os.makedirs(Agent.DEFAULT_MODELS_DIR_PATH, exist_ok=True)
        model_path = os.path.join(
            Agent.DEFAULT_MODELS_DIR_PATH, self.getModelName() + ".weights.h5"
        )
        self.model.save_weights(model_path)
        print(f"Model weights saved to {model_path}")
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
        return self.name + "Model"

    def getLogsName(self):
        return self.name + "Logs"

    def getMove(self, obs, info):
        move, frameInputs = self.getRandomMove(info)
        return move, frameInputs

    def initializeNetwork(self):
        raise NotImplementedError("Implement this in the inherited agent")

    def prepareMemoryForTraining(self, memory):
        raise NotImplementedError("Implement this in the inherited agent")

    def trainNetwork(self, data, model):
        raise NotImplementedError("Implement this in the inherited agent")


class DeepQAgent(Agent):
    EPSILON_MIN = 0.1
    DEFAULT_EPSILON_DECAY = 0.999
    DEFAULT_DISCOUNT_RATE = 0.98
    DEFAULT_LEARNING_RATE = 0.0001
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
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
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
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        if load and not resume:
            self.epsilon = DeepQAgent.EPSILON_MIN
        elif resume:
            self.epsilon = 0.9
        else:
            self.epsilon = epsilon
        self.epsilonDecay = DeepQAgent.DEFAULT_EPSILON_DECAY
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        super(DeepQAgent, self).__init__(load=load, name=name, moveList=moveList)

    def getMove(self, obs, info):
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs
        else:
            stateData = self.prepareNetworkInputs(info)
            with tf.device('/GPU:0'):
                predictedRewards = self.model.predict(stateData)[0]
            move = np.argmax(predictedRewards)
            frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
            return move, frameInputs

    def initializeNetwork(self):
        with tf.device('/GPU:0'):
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
        print("Successfully initialized model on GPU")
        return model

    def prepareMemoryForTraining(self, memory):
        data = []
        for step in self.memory:
            data.append(
                [
                    self.prepareNetworkInputs(step[Agent.STATE_INDEX]),
                    step[Agent.ACTION_INDEX],
                    step[Agent.REWARD_INDEX],
                    step[Agent.DONE_INDEX],
                    self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX]),
                ]
            )
        return data

    def prepareNetworkInputs(self, step):
        feature_vector = []
        feature_vector.append(step["enemy_health"])
        feature_vector.append(step["enemy_x_position"])
        feature_vector.append(step["enemy_y_position"])
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        if step["enemy_status"] not in DeepQAgent.doneKeys:
            oneHotEnemyState[DeepQAgent.stateIndices[step["enemy_status"]]] = 1
        feature_vector += oneHotEnemyState
        oneHotEnemyChar = [0] * 8
        oneHotEnemyChar[step["enemy_character"]] = 1
        feature_vector += oneHotEnemyChar
        feature_vector.append(step["health"])
        feature_vector.append(step["x_position"])
        feature_vector.append(step["y_position"])
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        if step["status"] not in DeepQAgent.doneKeys:
            oneHotPlayerState[DeepQAgent.stateIndices[step["status"]]] = 1
        feature_vector += oneHotPlayerState
        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def trainNetwork(self, data, model):
        max_samples = min(len(data), 100)
        minibatch = random.sample(data, max_samples)
        self.lossHistory.losses_clear()
        batch_count = 0
        max_batches = 20
        # Use GPU for training
        with tf.device('/GPU:0'):
            for state, action, reward, done, next_state in minibatch:
                if batch_count >= max_batches:
                    break
                modelOutput = model.predict(state)[0]
                if not done:
                    reward = reward + self.gamma * np.amax(model.predict(next_state)[0])
                modelOutput[action] = reward
                modelOutput = np.reshape(modelOutput, [1, self.actionSize])
                model.fit(
                    state, modelOutput, epochs=1, verbose=0, callbacks=[self.lossHistory]
                )
                batch_count += 1
        if self.epsilon > DeepQAgent.EPSILON_MIN:
            # so we will graduately stop learning new
            self.epsilon *= self.epsilonDecay
        return model


from keras.utils import get_custom_objects

get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})

# Print TensorFlow GPU info
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))
print("Devices:", tf.config.list_logical_devices())