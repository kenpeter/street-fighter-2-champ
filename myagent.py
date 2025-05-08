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
import json
import time
import pickle

print("TensorFlow version:", tf.__version__)
print("Is GPU available:", bool(tf.config.list_physical_devices("GPU")))
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    try:
        print(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled. Device list:", physical_devices)
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU devices found. Training will be slow on CPU only.")
    print("Make sure NVIDIA drivers and CUDA are properly installed.")

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
    DEFAULT_STATS_DIR_PATH = "../stats"

    def __init__(self, load=False, name=None, moveList=Moves):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.prepareForNextFight()
        self.moveList = moveList
        self.total_timesteps = 0
        self.episodes_completed = 0
        self.training_start_time = time.time()
        self.avg_reward_history = []
        self.avg_loss_history = []
        if self.__class__.__name__ != "Agent":
            self.model = self.initializeNetwork()
            if load:
                self.loadModel()
                self.loadStats()

    def prepareForNextFight(self):
        self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)
        self.episode_rewards = []

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
        self.total_timesteps += 1
        self.episode_rewards.append(step[Agent.REWARD_INDEX])

    def updateEpisodeMetrics(self):
        self.episodes_completed += 1
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.avg_reward_history.append(avg_reward)
            self.episode_rewards = []

    def reviewFight(self):
        data = self.prepareMemoryForTraining(self.memory)
        self.model = self.trainNetwork(data, self.model)
        self.updateEpisodeMetrics()
        if (hasattr(self, "lossHistory") and hasattr(self.lossHistory, "losses") and len(self.lossHistory.losses) > 0):
            avg_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
            self.avg_loss_history.append(avg_loss)
        if hasattr(self, "updateEpsilon"):
            self.updateEpsilon()
        self.saveModel()
        self.saveStats()
        self.printTrainingProgress()
        # to use the mem buffer
        # Removed: self.prepareForNextFight()


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

    def saveStats(self):
        os.makedirs(Agent.DEFAULT_STATS_DIR_PATH, exist_ok=True)
        stats_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_stats.json")
        memory_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_memory.pkl")  # New file for memory
        stats = {
            "total_timesteps": self.total_timesteps,
            "episodes_completed": self.episodes_completed,
            "avg_reward_history": self.avg_reward_history,
            "avg_loss_history": self.avg_loss_history,
            "saved_epsilon": self.epsilon,
        }
        try:
            with open(stats_path, "w") as file:
                json.dump(stats, file)
            with open(memory_path, "wb") as file:  # Save memory buffer
                pickle.dump(self.memory, file)
            print(f"Memory buffer saved to {memory_path}")
        except Exception as e:
            print(f"Error saving stats or memory: {e}")

    def loadStats(self):
        os.makedirs(Agent.DEFAULT_STATS_DIR_PATH, exist_ok=True)
        stats_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_stats.json")
        memory_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_memory.pkl")  # New file for memory
        self.loaded_stats = False
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as file:
                    stats = json.load(file)
                    self.total_timesteps = stats.get("total_timesteps", 0)
                    self.episodes_completed = stats.get("episodes_completed", 0)
                    self.avg_reward_history = stats.get("avg_reward_history", [])
                    self.avg_loss_history = stats.get("avg_loss_history", [])
                    self.saved_epsilon = stats.get("saved_epsilon", 0.9)
                    self.loaded_stats = True
                    print(f"Loaded training stats: {self.total_timesteps} timesteps completed over {self.episodes_completed} episodes")
                    print(f"Loaded saved epsilon value: {self.saved_epsilon}")
            except Exception as e:
                print(f"Error loading stats: {e}")
                print("Starting with fresh training statistics.")
        else:
            print("No previous training stats found. Starting fresh.")
        
        # Load memory buffer
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "rb") as file:
                    self.memory = pickle.load(file)
                    print(f"Loaded memory buffer from {memory_path} with {len(self.memory)} experiences")
            except Exception as e:
                print(f"Error loading memory buffer: {e}")
                self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)  # Fallback to empty deque
        else:
            self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)  # Initialize empty if no file
            print("No previous memory buffer found. Starting with empty buffer.")

    def printTrainingProgress(self):
        elapsed_time = time.time() - self.training_start_time
        print("\n==== Training Progress ====")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Episodes completed: {self.episodes_completed}")
        print(f"Training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            print(f"Recent average reward: {self.avg_reward_history[-1]:.4f}")
            if len(self.avg_reward_history) >= 2:
                reward_change = (
                    self.avg_reward_history[-1] - self.avg_reward_history[-2]
                )
                print(f"Reward change: {reward_change:+.4f}")
        if (
            hasattr(self, "lossHistory")
            and hasattr(self.lossHistory, "losses")
            and len(self.lossHistory.losses) > 0
        ):
            recent_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
            print(f"Recent loss: {recent_loss:.6f}")
            if self.avg_loss_history and len(self.avg_loss_history) >= 2:
                loss_change = self.avg_loss_history[-1] - self.avg_loss_history[-2]
                print(f"Loss change: {loss_change:+.6f}")
                if abs(loss_change) < 0.0001 and self.episodes_completed > 5:
                    print(
                        "WARNING: Training may be stuck in a local minimum - loss is not changing significantly"
                    )
                elif loss_change < 0:
                    print("Learning progress: Positive (loss is decreasing)")
                else:
                    print(
                        "Learning progress: Negative or stalled (loss is not decreasing)"
                    )
        print("===========================\n")

    def printFinalStats(self):
        elapsed_time = time.time() - self.training_start_time
        print("\n======= TRAINING SUMMARY =======")
        print(f"Total training timesteps: {self.total_timesteps}")
        print(f"Total episodes completed: {self.episodes_completed}")
        print(f"Total training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            print(f"Final average reward: {self.avg_reward_history[-1]:.4f}")
            if len(self.avg_reward_history) > 1:
                first_rewards = sum(self.avg_reward_history[:3]) / min(
                    3, len(self.avg_reward_history)
                )
                last_rewards = sum(self.avg_reward_history[-3:]) / min(
                    3, len(self.avg_reward_history)
                )
                reward_improvement = last_rewards - first_rewards
                print(f"Reward improvement: {reward_improvement:+.4f}")
        if self.avg_loss_history:
            print(f"Final average loss: {self.avg_loss_history[-1]:.6f}")
            if len(self.avg_loss_history) > 1:
                first_losses = sum(self.avg_loss_history[:3]) / min(
                    3, len(self.avg_loss_history)
                )
                last_losses = sum(self.avg_loss_history[-3:]) / min(
                    3, len(self.avg_loss_history)
                )
                loss_improvement = first_losses - last_losses
                print(f"Loss improvement: {loss_improvement:+.6f}")
                if loss_improvement > 0:
                    learning_status = "POSITIVE - Agent is learning effectively"
                elif loss_improvement > -0.001:
                    learning_status = "NEUTRAL - Small improvements in learning"
                else:
                    learning_status = (
                        "NEGATIVE - Agent may be stuck in suboptimal policy"
                    )
                print(f"Learning status: {learning_status}")
        print("=================================\n")

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
        epsilon=None,
        name=None,
        moveList=Moves,
    ):
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.total_timesteps = 0
        super(DeepQAgent, self).__init__(load=load, name=name, moveList=moveList)
        if epsilon is not None:
            self.epsilon = epsilon
            self.fixed_epsilon = True
        else:
            self.fixed_epsilon = False
            if load and not resume:
                self.epsilon = DeepQAgent.EPSILON_MIN
            else:
                self.calculateEpsilonFromTimesteps()
                print(
                    f"Epsilon set to {self.epsilon} based on {self.total_timesteps} total timesteps"
                )

    def calculateEpsilonFromTimesteps(self):
        START_EPSILON = 1.0
        TIMESTEPS_TO_MIN_EPSILON = 500000
        decay_per_step = (
            START_EPSILON - DeepQAgent.EPSILON_MIN
        ) / TIMESTEPS_TO_MIN_EPSILON
        self.epsilon = max(
            DeepQAgent.EPSILON_MIN,
            START_EPSILON - (decay_per_step * self.total_timesteps),
        )

    def updateEpsilon(self):
        if not self.fixed_epsilon:
            self.calculateEpsilonFromTimesteps()

    def getMove(self, obs, info):
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs
        else:
            stateData = self.prepareNetworkInputs(info)
            if len(tf.config.list_physical_devices("GPU")) > 0:
                with tf.device("/GPU:0"):
                    predictedRewards = self.model.predict(stateData)[0]
            else:
                predictedRewards = self.model.predict(stateData)[0]
            move = np.argmax(predictedRewards)
            frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
            return move, frameInputs

    def initializeNetwork(self):
        if len(tf.config.list_physical_devices("GPU")) > 0:
            with tf.device("/GPU:0"):
                model = Sequential()
                model.add(Dense(48, input_dim=self.stateSize, activation="relu"))
                model.add(Dense(96, activation="relu"))
                model.add(Dense(192, activation="relu"))
                model.add(Dense(96, activation="relu"))
                model.add(Dense(48, activation="relu"))
                model.add(Dense(self.actionSize, activation="linear"))
                model.compile(
                    loss=DeepQAgent._huber_loss,
                    optimizer=Adam(learning_rate=self.learningRate),
                )
                print("Successfully initialized model on GPU")
        else:
            model = Sequential()
            model.add(Dense(48, input_dim=self.stateSize, activation="relu"))
            model.add(Dense(96, activation="relu"))
            model.add(Dense(192, activation="relu"))
            model.add(Dense(96, activation="relu"))
            model.add(Dense(48, activation="relu"))
            model.add(Dense(self.actionSize, activation="linear"))
            model.compile(
                loss=DeepQAgent._huber_loss,
                optimizer=Adam(learning_rate=self.learningRate),
            )
            print("Successfully initialized model on CPU")
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
        if len(tf.config.list_physical_devices("GPU")) > 0:
            print("Training network on GPU...")
            with tf.device("/GPU:0"):
                for state, action, reward, done, next_state in minibatch:
                    if batch_count >= max_batches:
                        break
                    modelOutput = model.predict(state)[0]
                    if not done:
                        reward = reward + self.gamma * np.amax(
                            model.predict(next_state)[0]
                        )
                    modelOutput[action] = reward
                    modelOutput = np.reshape(modelOutput, [1, self.actionSize])
                    model.fit(
                        state,
                        modelOutput,
                        epochs=1,
                        verbose=0,
                        callbacks=[self.lossHistory],
                    )
                    batch_count += 1
        else:
            print("Training network on CPU...")
            for state, action, reward, done, next_state in minibatch:
                if batch_count >= max_batches:
                    break
                modelOutput = model.predict(state)[0]
                if not done:
                    reward = reward + self.gamma * np.amax(model.predict(next_state)[0])
                modelOutput[action] = reward
                modelOutput = np.reshape(modelOutput, [1, self.actionSize])
                model.fit(
                    state,
                    modelOutput,
                    epochs=1,
                    verbose=0,
                    callbacks=[self.lossHistory],
                )
                batch_count += 1
        return model

from keras.utils import get_custom_objects
get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})

print("\nGPU configuration summary:")
print("==========================")
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("GPU device name:", tf.test.gpu_device_name())
else:
    print("No GPU found. Using CPU only.")
print("==========================\n")