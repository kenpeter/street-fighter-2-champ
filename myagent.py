import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Lambda, Input, Add
from keras.optimizers import Adam
from keras import backend as K
import keras.losses
from DefaultMoveList import Moves
from LossHistory import LossHistory
import json
import time
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Agent")

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

# Register custom Huber loss function
def _huber_loss(y_true, y_pred, clip_delta=1.0):
    import tensorflow as tf
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (
        tf.abs(error) - clip_delta
    )
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

from keras.utils import get_custom_objects
get_custom_objects().update({"_huber_loss": _huber_loss})

class CircularBuffer:
    """A circular buffer implementation for storing experiences efficiently."""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.current_index = 0
        self.size = 0
    
    def append(self, item):
        self.buffer[self.current_index] = item
        self.current_index = (self.current_index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, weights=None):
        """Sample elements from buffer with optional weights for prioritization"""
        if weights is not None:
            indices = random.choices(range(self.size), weights=weights[:self.size], k=min(batch_size, self.size))
        else:
            indices = random.sample(range(self.size), min(batch_size, self.size))
        return [self.buffer[i] for i in indices], indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for specific indices"""
        for idx, priority in zip(indices, priorities):
            if self.buffer[idx] is not None:
                self.buffer[idx][-1] = priority
    
    def get_all(self):
        """Return all valid entries in the buffer"""
        return [self.buffer[i] for i in range(self.size) if self.buffer[i] is not None]
    
    def __len__(self):
        return self.size

class Agent:
    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    PRIORITY_INDEX = 7
    MAX_DATA_LENGTH = 200000
    DEFAULT_MODELS_DIR_PATH = "./models"
    DEFAULT_LOGS_DIR_PATH = "./logs"
    DEFAULT_STATS_DIR_PATH = "./stats"

    def __init__(self, resume=False, name=None, moveList=Moves):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
            
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        Agent.DEFAULT_MODELS_DIR_PATH = os.path.join(self.current_dir, "models")
        Agent.DEFAULT_LOGS_DIR_PATH = os.path.join(self.current_dir, "logs")
        Agent.DEFAULT_STATS_DIR_PATH = os.path.join(self.current_dir, "stats")
        
        os.makedirs(Agent.DEFAULT_MODELS_DIR_PATH, exist_ok=True)
        os.makedirs(Agent.DEFAULT_LOGS_DIR_PATH, exist_ok=True)
        os.makedirs(Agent.DEFAULT_STATS_DIR_PATH, exist_ok=True)
        
        logger.info(f"Agent {self.name} initialization:")
        logger.info(f"Models directory: {Agent.DEFAULT_MODELS_DIR_PATH}")
        logger.info(f"Logs directory: {Agent.DEFAULT_LOGS_DIR_PATH}")
        logger.info(f"Stats directory: {Agent.DEFAULT_STATS_DIR_PATH}")
        
        self.prepareForNextFight()
        self.moveList = moveList
        self.total_timesteps = 0
        self.episodes_completed = 0
        self.training_start_time = time.time()
        self.avg_reward_history = []
        self.avg_loss_history = []

        self.lr_step_size = 50000
        self.last_lr_update = 0
        
        if self.__class__.__name__ != "Agent":
            self.model = self.initializeNetwork()
            if resume:
                self.loadModel()
                self.loadStats()
                logger.info(f"Resumed training for {self.name} from existing model")
            else:
                logger.info(f"Starting fresh training for {self.name}")

    def prepareForNextFight(self):
        self.memory = CircularBuffer(Agent.MAX_DATA_LENGTH)
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
        modified_step = list(step)
        
        current_enemy_health = step[Agent.STATE_INDEX].get("enemy_health", 100)
        current_player_health = step[Agent.STATE_INDEX].get("health", 100)
        next_enemy_health = step[Agent.NEXT_STATE_INDEX].get("enemy_health", 100)
        next_player_health = step[Agent.NEXT_STATE_INDEX].get("health", 100)
        
        enemy_damage = max(0, current_enemy_health - next_enemy_health)
        player_damage = max(0, current_player_health - next_player_health)
        
        reward = modified_step[Agent.REWARD_INDEX]
        
        health_reward = player_damage * -0.2
        enemy_health_reward = enemy_damage * 0.3
        
        if next_enemy_health <= 0:
            reward += 50
        if next_player_health <= 0:
            reward -= 25
            
        x_distance = abs(step[Agent.STATE_INDEX].get("x_position", 0) - 
                         step[Agent.STATE_INDEX].get("enemy_x_position", 0))
        
        player_attacking = (current_player_health > 50 and current_enemy_health < 50)
        position_reward = 0
        if player_attacking and x_distance < 50:
            position_reward = 0.1
        elif not player_attacking and x_distance > 100:
            position_reward = 0.1
        
        total_reward = reward + health_reward + enemy_health_reward + position_reward
        
        modified_step[Agent.REWARD_INDEX] = total_reward
        
        priority = abs(total_reward) + 0.01
        if next_enemy_health <= 0:
            priority *= 3.0
        if enemy_damage > 10:
            priority *= 2.0
        
        modified_step_with_priority = modified_step + [priority]
        self.memory.append(modified_step_with_priority)
        self.total_timesteps += 1
        self.episode_rewards.append(total_reward)

    def updateEpisodeMetrics(self):
        self.episodes_completed += 1
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.avg_reward_history.append(avg_reward)
            self.episode_rewards = []

    def reviewFight(self):
        self.episode_count += 1
        
        if self.episode_count % self.update_target_every == 0:
            self.target_model.set_weights(self.model.get_weights())
            logger.info(f"Target network updated at episode {self.episode_count}")
        
        try:
            data = self.prepareMemoryForTraining(self.memory)
            if len(data) > 0:
                temp_lr = self.model.optimizer.learning_rate.numpy() * 0.5
                self.model.optimizer.learning_rate.assign(temp_lr)
                self.model = self.trainNetwork(data, self.model)
                self.model.optimizer.learning_rate.assign(temp_lr * 2)
        except Exception as e:
            logger.error(f"Error training network: {e}")
        
        try:
            self.updateEpisodeMetrics()
            if hasattr(self, "lossHistory") and hasattr(self.lossHistory, "losses") and len(self.lossHistory.losses) > 0:
                avg_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
                self.avg_loss_history.append(avg_loss)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        if hasattr(self, "updateEpsilon"):
            try:
                self.updateEpsilon()
            except Exception as e:
                logger.error(f"Error updating epsilon: {e}")
        
        try:
            current_time = self.total_timesteps
            if current_time - self.last_lr_update >= self.lr_step_size:
                current_lr = self.model.optimizer.learning_rate.numpy()
                new_lr = current_lr * self.lr_decay
                self.model.optimizer.learning_rate.assign(new_lr)
                logger.info(f"Learning rate decayed to {new_lr:.6f} at {current_time} timesteps")
                self.last_lr_update = current_time
        except Exception as e:
            logger.error(f"Error applying learning rate decay: {e}")
        
        try:
            self.saveModel()
            self.saveStats()
            self.printTrainingProgress()
        except Exception as e:
            logger.error(f"Error saving or printing: {e}")

    def loadModel(self):
        models_dir = os.path.join(self.current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{self.getModelName()}")
        load_successful = False
        for ext in [".weights.h5", ".h5", ".keras"]:
            full_path = model_path + ext
            if os.path.exists(full_path):
                try:
                    self.model.load_weights(full_path)
                    load_successful = True
                    logger.info(f"✓ Model loaded from {full_path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {full_path}: {e}")
        if load_successful and hasattr(self, "target_model"):
            self.target_model.set_weights(self.model.get_weights())
            logger.info("Target network synchronized")
        if not load_successful:
            logger.warning("No valid model file found, using new model.")

    def saveModel(self):
        models_dir = os.path.join(self.current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{self.getModelName()}.weights.h5")
        try:
            self.model.save_weights(model_path)
            logger.info(f"✓ Model weights saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def saveStats(self):
        stats_dir = os.path.join(self.current_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{self.name}_stats.json")
        memory_path = os.path.join(stats_dir, f"{self.name}_memory.pkl")
        stats = {
            "total_timesteps": self.total_timesteps,
            "episodes_completed": self.episodes_completed,
            "avg_reward_history": self.avg_reward_history,
            "avg_loss_history": self.avg_loss_history,
            "last_lr_update": getattr(self, "last_lr_update", 0),
            "episode_count": getattr(self, "episode_count", 0),
        }
        try:
            with open(stats_path, "w") as file:
                json.dump(stats, file, indent=4)
            with open(memory_path, "wb") as file:
                memory_data = self.memory.get_all()
                pickle.dump(memory_data, file)
            logger.info(f"✓ Stats and memory saved")
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

    def loadStats(self):
        stats_dir = os.path.join(self.current_dir, "stats")
        stats_path = os.path.join(stats_dir, f"{self.name}_stats.json")
        memory_path = os.path.join(stats_dir, f"{self.name}_memory.pkl")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as file:
                    stats = json.load(file)
                    self.total_timesteps = stats.get("total_timesteps", 0)
                    self.episodes_completed = stats.get("episodes_completed", 0)
                    self.avg_reward_history = stats.get("avg_reward_history", [])
                    self.avg_loss_history = stats.get("avg_loss_history", [])
                    self.last_lr_update = stats.get("last_lr_update", 0)
                    if hasattr(self, "episode_count"):
                        self.episode_count = stats.get("episode_count", 0)
                    if hasattr(self, "calculateEpsilonFromTimesteps"):
                        self.calculateEpsilonFromTimesteps()
                    logger.info(f"✓ Loaded stats from {stats_path}")
            except Exception as e:
                logger.error(f"Error loading stats: {e}")
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "rb") as file:
                    memory_data = pickle.load(file)
                    self.memory = CircularBuffer(Agent.MAX_DATA_LENGTH)
                    for experience in memory_data:
                        self.memory.append(experience)
                    logger.info(f"✓ Loaded memory with {len(memory_data)} experiences")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

    def printTrainingProgress(self):
        elapsed_time = time.time() - self.training_start_time
        logger.info("\n==== Training Progress ====")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        logger.info(f"Episodes completed: {self.episodes_completed}")
        logger.info(f"Training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            logger.info(f"Recent average reward: {self.avg_reward_history[-1]:.4f}")
        if hasattr(self, "lossHistory") and hasattr(self.lossHistory, "losses") and len(self.lossHistory.losses) > 0:
            recent_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
            logger.info(f"Recent loss: {recent_loss:.6f}")
        logger.info("===========================\n")

    def printFinalStats(self):
        elapsed_time = time.time() - self.training_start_time
        logger.info("\n======= TRAINING SUMMARY =======")
        logger.info(f"Total training timesteps: {self.total_timesteps}")
        logger.info(f"Total episodes completed: {self.episodes_completed}")
        logger.info(f"Total training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            logger.info(f"Final average reward: {self.avg_reward_history[-1]:.4f}")
        if self.avg_loss_history:
            logger.info(f"Final average loss: {self.avg_loss_history[-1]:.6f}")
        logger.info("=================================\n")

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
        experiences = memory.get_all()
        if not experiences:
            return []
            
        action_counts = {}
        for step in experiences:
            action = step[Agent.ACTION_INDEX]
            action_counts[action] = action_counts.get(action, 0) + 1
            
        data_with_priority = []
        beta = 1.0
        
        for step in experiences:
            state = self.prepareNetworkInputs(step[Agent.STATE_INDEX])
            action = step[Agent.ACTION_INDEX]
            reward = step[Agent.REWARD_INDEX]
            done = step[Agent.DONE_INDEX]
            next_state = self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX])
            action_frequency = action_counts[action] / len(experiences)
            rarity_score = beta * (1.0 - action_frequency)
            priority = abs(reward) + rarity_score
            data_with_priority.append([state, action, reward, done, next_state, priority])
        
        data_with_priority.sort(key=lambda x: x[5], reverse=True)
        total_size = len(data_with_priority)
        top_percent = int(0.7 * total_size)
        top_experiences = data_with_priority[:top_percent]
        remaining = data_with_priority[top_percent:]
        random_count = min(int(0.3 * total_size), len(remaining))
        random_experiences = random.sample(remaining, random_count) if random_count > 0 and remaining else []
        final_data = top_experiences + random_experiences
        return final_data

    def prepareNetworkInputs(self, step):
        feature_vector = []
        player_health = step.get("health", 100)
        enemy_health = step.get("enemy_health", 100)
        feature_vector.append(player_health)
        feature_vector.append(enemy_health)
        player_health_pct = player_health / 100.0
        enemy_health_pct = enemy_health / 100.0
        health_advantage = player_health_pct - enemy_health_pct
        feature_vector.append(player_health_pct)
        feature_vector.append(enemy_health_pct)
        feature_vector.append(health_advantage)
        player_x = step.get("x_position", 0)
        player_y = step.get("y_position", 0)
        enemy_x = step.get("enemy_x_position", 0)
        enemy_y = step.get("enemy_y_position", 0)
        feature_vector.append(player_x)
        feature_vector.append(player_y)
        feature_vector.append(enemy_x)
        feature_vector.append(enemy_y)
        x_distance = abs(player_x - enemy_x)
        y_distance = abs(player_y - enemy_y)
        euclidean_distance = np.sqrt(x_distance**2 + y_distance**2)
        feature_vector.append(x_distance)
        feature_vector.append(y_distance)
        feature_vector.append(euclidean_distance)
        facing_right = 1 if player_x < enemy_x else 0
        feature_vector.append(facing_right)
        enemy_above = 1 if player_y > enemy_y else 0
        feature_vector.append(enemy_above)
        enemy_status = step.get("enemy_status", 512)
        player_status = step.get("status", 512)
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(enemy_status, 0)
        oneHotEnemyState[state_index] = 1
        feature_vector += oneHotEnemyState
        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector += oneHotEnemyChar
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(player_status, 0)
        oneHotPlayerState[state_index] = 1
        feature_vector += oneHotPlayerState
        
        if len(feature_vector) != self.stateSize:
            if len(feature_vector) < self.stateSize:
                feature_vector += [0] * (self.stateSize - len(feature_vector))
            else:
                feature_vector = feature_vector[:self.stateSize]
        
        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def trainNetwork(self, data, model):
        if len(data) == 0:
            logger.warning("No data available for training. Skipping.")
            return model
            
        minibatch, _ = self.memory.sample(64)
        states = []
        targets = []
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        
        with tf.device(device):
            for step in minibatch:
                state = step[Agent.STATE_INDEX]
                action = step[Agent.ACTION_INDEX]
                reward = step[Agent.REWARD_INDEX]
                next_state = step[Agent.NEXT_STATE_INDEX]
                done = step[Agent.DONE_INDEX]
                if action >= self.actionSize:
                    action = action % self.actionSize
                state_data = self.prepareNetworkInputs(state)
                q_values = model.predict(state_data, verbose=0)[0]
                if len(q_values.shape) > 1:
                    q_values = q_values.flatten()
                if len(q_values) != self.actionSize:
                    q_values = np.ones(self.actionSize) / self.actionSize
                target = q_values.copy()
                if done:
                    target[action] = reward
                else:
                    next_state_data = self.prepareNetworkInputs(next_state)
                    next_q_values = self.target_model.predict(next_state_data, verbose=0)[0]
                    if len(next_q_values.shape) > 1:
                        next_q_values = next_q_values.flatten()
                    if len(next_q_values) != self.actionSize:
                        next_q_values = np.ones(self.actionSize) / self.actionSize
                    best_action = np.argmax(model.predict(next_state_data, verbose=0)[0])
                    target[action] = reward + self.gamma * next_q_values[best_action]
                states.append(state_data[0])
                targets.append(target)
            
            states = np.array(states)
            targets = np.array(targets)
            model.fit(
                states,
                targets,
                batch_size=64,
                epochs=1,
                verbose=0,
                validation_split=0.2,
                callbacks=[self.lossHistory]
            )
        return model

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

class DeepQAgent(Agent):
    EPSILON_MIN = 0.05
    DEFAULT_DISCOUNT_RATE = 0.98
    DEFAULT_LEARNING_RATE = 0.0001
    stateIndices = {
        512: 0, 514: 1, 516: 2, 518: 3, 520: 4, 522: 5, 524: 6, 526: 7, 532: 8,
    }
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]

    def __init__(self, stateSize=32, resume=False, name=None, moveList=Moves):
        self.stateSize = stateSize
        self.actionSize = len(moveList) if hasattr(moveList, '__len__') else 20
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.total_timesteps = 0
        self.episode_count = 0
        self.update_target_every = 2
        self.lr_decay = 0.995
        
        super(DeepQAgent, self).__init__(resume=resume, name=name, moveList=moveList)
        
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())
        
        if resume and hasattr(self, "total_timesteps"):
            self.calculateEpsilonFromTimesteps()
            logger.info(f"Resuming with epsilon: {self.epsilon}")
        else:
            self.epsilon = 1.0
            logger.info(f"Starting with epsilon: {self.epsilon}")

    def recordStep(self, step):
        modified_step = list(step)
        current_state = step[Agent.STATE_INDEX]
        next_state = step[Agent.NEXT_STATE_INDEX]
        reward = step[Agent.REWARD_INDEX]
        
        current_player_health = current_state.get("health", 100)
        current_enemy_health = current_state.get("enemy_health", 100)
        next_player_health = next_state.get("health", 100)
        next_enemy_health = next_state.get("enemy_health", 100)
        
        damage_dealt = max(0, current_enemy_health - next_enemy_health)
        damage_taken = max(0, current_player_health - next_player_health)
        
        player_x = current_state.get("x_position", 0)
        enemy_x = current_state.get("enemy_x_position", 0)
        distance = abs(player_x - enemy_x)
        
        combo_count = current_state.get("combo_count", 0)
        
        # Damage rewards
        damage_reward = damage_dealt * 0.2
        damage_penalty = damage_taken * -0.2
        modified_reward = reward + damage_reward + damage_penalty
        
        # Victory and defeat
        if next_enemy_health <= 0:
            modified_reward += 150 + (next_player_health / 100 * 50)
        elif next_player_health <= 0:
            modified_reward -= 75
        
        # Combo bonus
        combo_bonus = combo_count * 3
        modified_reward += combo_bonus
        
        # Positional advantage
        if distance < 50:
            modified_reward += 0.1 * (50 - distance)
        
        modified_step[Agent.REWARD_INDEX] = modified_reward
        
        priority = abs(modified_reward) + 0.01
        if next_enemy_health <= 0:
            priority *= 8.0
        if damage_dealt > 10:
            priority *= 2.0
        
        modified_step_with_priority = modified_step + [priority]
        self.memory.append(modified_step_with_priority)
        self.total_timesteps += 1
        self.episode_rewards.append(modified_reward)

    def calculateEpsilonFromTimesteps(self):
        START_EPSILON = 1.0
        TIMESTEPS_TO_MIN_EPSILON = 750000  # Note: Should be 1500000 for 50% longer than 1000000, assuming typo
        decay_per_step = (START_EPSILON - DeepQAgent.EPSILON_MIN) / TIMESTEPS_TO_MIN_EPSILON
        self.epsilon = max(DeepQAgent.EPSILON_MIN, START_EPSILON - (decay_per_step * self.total_timesteps))
        
        if self.episodes_completed > 20:
            win_rate = sum(1 for r in self.episode_rewards[-100:] if r > 0) / min(len(self.episode_rewards), 100) if self.episode_rewards else 0
            if win_rate < 0.15:
                self.epsilon = min(0.8, self.epsilon + 0.15)
                logger.info(f"Performance boost: epsilon increased to {self.epsilon}")
        
        self.epsilon *= 0.995

    def updateEpsilon(self):
        old_epsilon = self.epsilon
        self.calculateEpsilonFromTimesteps()
        if abs(old_epsilon - self.epsilon) > 0.01:
            logger.info(f"Epsilon updated: {old_epsilon:.4f} -> {self.epsilon:.4f}")

    def getMove(self, obs, info):
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs
        else:
            stateData = self.prepareNetworkInputs(info)
            device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
            try:
                with tf.device(device):
                    predictions = self.model.predict(stateData, verbose=0)[0]
                    if len(predictions.shape) > 1:
                        predictions = predictions.flatten()
                    if len(predictions) != self.actionSize:
                        predictions = np.ones(self.actionSize) / self.actionSize
                    move = np.argmax(predictions)
                    if move >= len(self.moveList):
                        move = move % len(self.moveList)
                    frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
                    return move, frameInputs
            except Exception as e:
                logger.error(f"Error in getMove: {e}")
                return self.getRandomMove(info)

    def initializeNetwork(self):
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        with tf.device(device):
            input_layer = Input(shape=(self.stateSize,))
            x = BatchNormalization()(input_layer)
            
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            residual = x
            
            x = Dense(256, activation='linear')(x)
            x = Add()([residual, x])
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            
            x = Dense(64, activation='relu')(x)
            
            value_stream = Dense(32, activation='relu')(x)
            value = Dense(1)(value_stream)
            
            advantage_stream = Dense(32, activation='relu')(x)
            advantage = Dense(self.actionSize)(advantage_stream)
            
            q_values = Lambda(lambda a: a[0] + (a[1] - tf.math.reduce_mean(a[1], axis=1, keepdims=True)),
                              output_shape=(self.actionSize,))([value, advantage])
            
            model = Model(inputs=input_layer, outputs=q_values)
            optimizer = Adam(learning_rate=0.00015, clipnorm=1.0)
            model.compile(loss=_huber_loss, optimizer=optimizer)
            params = sum([np.prod(w.shape) for w in model.trainable_weights])
            logger.info(f"Initialized DQN model with residual connections (~{params:,} parameters)")
        return model

    def prepareMemoryForTraining(self, memory):
        experiences = memory.get_all()
        if not experiences:
            return []
        
        data_with_priority = []
        for step in experiences:
            state = self.prepareNetworkInputs(step[Agent.STATE_INDEX])
            action = step[Agent.ACTION_INDEX]
            reward = step[Agent.REWARD_INDEX]
            done = step[Agent.DONE_INDEX]
            next_state = self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX])
            priority = abs(reward) + 0.01
            
            if step[Agent.NEXT_STATE_INDEX].get('enemy_health', 100) <= 0:
                priority *= 8.0
            steps_survived = step[Agent.NEXT_STATE_INDEX].get('match_duration', 0) - step[Agent.STATE_INDEX].get('match_duration', 0)
            priority += steps_survived * 0.01
            
            data_with_priority.append([state, action, reward, done, next_state, priority])
        
        data_with_priority.sort(key=lambda x: x[5], reverse=True)
        total_size = len(data_with_priority)
        high_priority_size = int(0.5 * total_size)
        recent_size = int(0.3 * total_size)
        random_size = total_size - high_priority_size - recent_size
        
        high_priority = data_with_priority[:high_priority_size]
        recent = data_with_priority[-recent_size:] if recent_size > 0 else []
        remaining = data_with_priority[high_priority_size:-recent_size] if recent_size > 0 else data_with_priority[high_priority_size:]
        random_sample = random.sample(remaining, min(random_size, len(remaining))) if remaining and random_size > 0 else []
        
        return high_priority + recent + random_sample

    def trainNetwork(self, data, model):
        if len(data) == 0:
            return model
        
        batch_size = 64
        indices = random.sample(range(len(data)), min(batch_size, len(data)))
        minibatch = [data[i] for i in indices]
        
        states = []
        targets = []
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        
        with tf.device(device):
            with tf.GradientTape() as tape:
                for step in minibatch:
                    state = step[Agent.STATE_INDEX]
                    action = step[Agent.ACTION_INDEX]
                    reward = step[Agent.REWARD_INDEX]
                    done = step[Agent.DONE_INDEX]
                    next_state = step[Agent.NEXT_STATE_INDEX]
                    
                    if action >= self.actionSize:
                        action = action % self.actionSize
                    
                    q_values = model(state, training=False)[0]
                    if len(q_values.shape) > 1:
                        q_values = q_values.flatten()
                    if len(q_values) != self.actionSize:
                        q_values = np.ones(self.actionSize) / self.actionSize
                    
                    target = q_values.numpy().copy()
                    if done:
                        target[action] = reward
                    else:
                        next_q_values = self.target_model(next_state, training=False)[0]
                        if len(next_q_values.shape) > 1:
                            next_q_values = next_q_values.flatten()
                        if len(next_q_values) != self.actionSize:
                            next_q_values = np.ones(self.actionSize) / self.actionSize
                        best_action = np.argmax(model(next_state, training=False)[0])
                        target[action] = reward + self.gamma * next_q_values[best_action]
                    
                    states.append(state[0])
                    targets.append(target)
                
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                targets = tf.convert_to_tensor(targets, dtype=tf.float32)
                
                predictions = model(states, training=True)
                loss = _huber_loss(targets, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if len(self.avg_loss_history) >= 2 and abs(self.avg_loss_history[-1] - self.avg_loss_history[-2]) > 2.0:
            self.target_model.set_weights(model.get_weights())
            logger.info("Emergency target network update due to loss spike")
        
        return model