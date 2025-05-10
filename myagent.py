import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
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
    MAX_DATA_LENGTH = 500000  # Requirement 1: Increased from 50000 to 200000 (4x)
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

        self.lr_step_size = 10000  # Apply decay every 10,000 timesteps
        self.last_lr_update = 0  # Timestep of last update

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
        memory_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_memory.pkl")
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
            with open(memory_path, "wb") as file:
                pickle.dump(self.memory, file)
            print(f"Memory buffer saved to {memory_path}")
        except Exception as e:
            print(f"Error saving stats or memory: {e}")

    def loadStats(self):
        os.makedirs(Agent.DEFAULT_STATS_DIR_PATH, exist_ok=True)
        stats_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_stats.json")
        memory_path = os.path.join(Agent.DEFAULT_STATS_DIR_PATH, f"{self.name}_memory.pkl")
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
        
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "rb") as file:
                    self.memory = pickle.load(file)
                    print(f"Loaded memory buffer from {memory_path} with {len(self.memory)} experiences")
            except Exception as e:
                print(f"Error loading memory buffer: {e}")
                self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)
        else:
            self.memory = deque(maxlen=Agent.MAX_DATA_LENGTH)
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
        self.episode_count = 0  # For target network updates
        self.update_target_every = 5  # Requirement 2: Update target network every 5 episodes
        self.lr_decay = 0.995  # Requirement 4: Learning rate decay factor
        super(DeepQAgent, self).__init__(load=load, name=name, moveList=moveList)
        # Requirement 2: Initialize target network
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())
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
            with tf.device("/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"):
                predictedRewards = self.model.predict(stateData)[0]
            move = np.argmax(predictedRewards)
            frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
            return move, frameInputs

    def initializeNetwork(self):
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        with tf.device(device):
            model = Sequential()
            model.add(Dense(256, input_dim=self.stateSize))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(256))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(256))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.actionSize, activation='linear'))
            model.compile(
                loss=DeepQAgent._huber_loss,
                optimizer=Adam(learning_rate=self.learningRate),
            )
            print(f"Successfully initialized model on {device}")
        return model


    def recordStep(self, step):
        # Append step with initial priority based on reward
        priority = abs(step[Agent.REWARD_INDEX])
        self.memory.append(list(step) + [priority])
        self.total_timesteps += 1
        self.episode_rewards.append(step[Agent.REWARD_INDEX])

    def prepareMemoryForTraining(self, memory):
        # Requirement 5: Calculate action rarity, diversity, and rarity bonuses
        action_counts = {}
        for step in memory:
            action = step[Agent.ACTION_INDEX]
            action_counts[action] = action_counts.get(action, 0) + 1

        # Compute priorities for all experiences
        data_with_priority = []
        beta = 1.0  # Hyperparameter for action rarity weight
        for step in memory:
            state = self.prepareNetworkInputs(step[Agent.STATE_INDEX])
            action = step[Agent.ACTION_INDEX]
            reward = step[Agent.REWARD_INDEX]
            done = step[Agent.DONE_INDEX]
            next_state = self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX])
            # Action rarity: inverse frequency
            rarity_score = beta / (action_counts[action] + 1)
            # Basic priority: |reward| + rarity bonus (diversity and state rarity simplified)
            priority = abs(reward) + rarity_score
            data_with_priority.append([state, action, reward, done, next_state, priority])

        # Sort by priority and select top 70%
        data_with_priority.sort(key=lambda x: x[5], reverse=True)
        top_70_percent = int(0.7 * len(data_with_priority))
        top_data = data_with_priority[:top_70_percent]
        remaining_data = data_with_priority[top_70_percent:]

        # Ensure diversity in remaining 30% by selecting unique states
        selected_remaining = []
        seen_states = set()
        for item in top_data:
            state_tuple = tuple(item[0].flatten())
            seen_states.add(state_tuple)

        # From remaining, pick experiences with unique states
        for item in remaining_data:
            state_tuple = tuple(item[0].flatten())
            if state_tuple not in seen_states:
                selected_remaining.append(item)
                seen_states.add(state_tuple)
            if len(selected_remaining) >= int(0.3 * len(data_with_priority)):
                break

        # If not enough unique states, fill with remaining data randomly
        if len(selected_remaining) < int(0.3 * len(data_with_priority)):
            needed = int(0.3 * len(data_with_priority)) - len(selected_remaining)
            available = [x for x in remaining_data if tuple(x[0].flatten()) not in seen_states]
            if available and needed > 0:
                extra = random.sample(available, min(needed, len(available)))
                selected_remaining.extend(extra)

        # Combine top 70% and selected remaining
        final_data = top_data + selected_remaining
        return [(i, d[0], d[1], d[2], d[3], d[4], d[5]) for i, d in enumerate(final_data)]

    def prepareNetworkInputs(self, step):
        feature_vector = []
        # e health
        feature_vector.append(step["enemy_health"])
        # e x
        feature_vector.append(step["enemy_x_position"])
        # e y
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
        # Requirement 6 & 7: Use TD error for prioritization, increase max_batches to 50
        if len(data) == 0:
            return model
        # Extract priorities and sample minibatch
        priorities = [d[6] for d in data]
        total_priority = sum(priorities)
        probabilities = [p / total_priority if total_priority > 0 else 1.0 / len(data) for p in priorities]
        max_samples = min(len(data), 256)  # Increased for better GPU utilization
        minibatch_indices = random.choices(range(len(data)), weights=probabilities, k=max_samples)
        minibatch = [data[i] for i in minibatch_indices]

        states = []
        targets = []
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        with tf.device(device):
            for idx, state, action, reward, done, next_state, _ in minibatch:
                modelOutput = model.predict(state)[0]
                if not done:
                    # Requirement 2: Use target network for stability
                    target_next = self.target_model.predict(next_state)[0]
                    target_q = reward + self.gamma * np.amax(target_next)
                else:
                    target_q = reward
                target = modelOutput.copy()
                target[action] = target_q
                states.append(state[0])
                targets.append(target)
                # Requirement 6: Compute TD error for prioritization
                predicted = modelOutput[action]
                td_error = abs(target_q - predicted)
                # Update priority in memory (memory stores step + priority)
                memory_idx = self.memory[idx].index
                self.memory[memory_idx][7] = td_error

            states = np.array(states)
            targets = np.array(targets)
            # Requirement 7: Train on larger batch (max_batches concept integrated into single fit)
            model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.lossHistory])
        return model

    def reviewFight(self):
        self.episode_count += 1
        # Requirement 2: Update target network infrequently
        if self.episode_count % self.update_target_every == 0:
            self.target_model.set_weights(self.model.get_weights())
            print(f"Target network updated at episode {self.episode_count}")
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


        # Then replace the learning rate decay code in reviewFight() with:
        if self.total_timesteps - self.last_lr_update >= self.lr_step_size:
            current_lr = K.get_value(self.model.optimizer.lr)
            new_lr = current_lr * self.lr_decay
            K.set_value(self.model.optimizer.lr, new_lr)
            print(f"Learning rate decayed to {new_lr} at {self.total_timesteps} timesteps")
            self.last_lr_update = self.total_timesteps


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