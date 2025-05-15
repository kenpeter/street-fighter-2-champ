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
        # FIX #4: Add TD error tracking for proper prioritization
        self.priorities = np.zeros(max_size, dtype=np.float32)
    
    def append(self, item):
        # FIX #4: Initialize with max priority for new experiences
        new_priority = 1.0  # Max priority for new experiences
        if self.size > 0:
            new_priority = np.max(self.priorities[:self.size])
        
        self.buffer[self.current_index] = item
        self.priorities[self.current_index] = new_priority
        self.current_index = (self.current_index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, use_priorities=True):
        """Sample elements from buffer with prioritization"""
        if self.size == 0:
            return [], []
            
        if use_priorities and np.sum(self.priorities[:self.size]) > 0:
            # FIX #4: Use proper prioritized sampling
            probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])
            indices = np.random.choice(range(self.size), min(batch_size, self.size), 
                                      p=probs, replace=False)
        else:
            indices = random.sample(range(self.size), min(batch_size, self.size))
            
        return [self.buffer[i] for i in indices], indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        # FIX #4: Use TD error for prioritization with offset to ensure non-zero probability
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < self.size and self.buffer[idx] is not None:
                self.priorities[idx] = abs(td_error) + 1e-5
    
    def get_all(self):
        """Return all valid entries in the buffer"""
        return [item.copy() if isinstance(item, list) else item for item in self.buffer[:self.size] if item is not None]
    
    def __len__(self):
        return self.size

class DeepQAgent:
    """Deep Q-Learning agent with Double DQN and Dueling architecture"""
    
    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    PRIORITY_INDEX = 7
    MAX_DATA_LENGTH = 200000

    EPSILON_MIN = 0.05
    DEFAULT_DISCOUNT_RATE = 0.98
    DEFAULT_LEARNING_RATE = 0.0001
    stateIndices = {
        512: 0, 514: 1, 516: 2, 518: 3, 520: 4, 
        522: 5, 524: 6, 526: 7, 532: 8,
    }
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]
    ACTION_BUTTONS = ["X", "Y", "Z", "A", "B", "C"]

    def __init__(self, stateSize=45, resume=False, name=None, moveList=Moves):
        """
        Initialize the DeepQAgent
        
        Args:
            stateSize: Size of the state vector (increased to 45 for all features)
            resume: Whether to resume training from saved model
            name: Name of the agent
            moveList: List of available moves
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
            
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.DEFAULT_MODELS_DIR_PATH = os.path.join(self.current_dir, "models")
        self.DEFAULT_LOGS_DIR_PATH = os.path.join(self.current_dir, "logs")
        self.DEFAULT_STATS_DIR_PATH = os.path.join(self.current_dir, "stats")
        
        os.makedirs(self.DEFAULT_MODELS_DIR_PATH, exist_ok=True)
        os.makedirs(self.DEFAULT_LOGS_DIR_PATH, exist_ok=True)
        os.makedirs(self.DEFAULT_STATS_DIR_PATH, exist_ok=True)
        
        logger.info(f"Agent {self.name} initialization:")
        logger.info(f"Models directory: {self.DEFAULT_MODELS_DIR_PATH}")
        logger.info(f"Logs directory: {self.DEFAULT_LOGS_DIR_PATH}")
        logger.info(f"Stats directory: {self.DEFAULT_STATS_DIR_PATH}")
        
        self.prepareForNextFight()
        self.moveList = moveList
        self.total_timesteps = 0
        self.episodes_completed = 0
        self.training_start_time = time.time()
        self.avg_reward_history = []
        self.avg_loss_history = []
        self.episode_outcomes = []  # Track win/loss outcomes
        
        self.lr_step_size = 50000
        self.last_lr_update = 0
        self.lr_decay = 0.995
        
        # FIX #1: Increase stateSize to accommodate all features (5+7+2+3+9+8+9 = 43, +2 buffer = 45)
        self.stateSize = stateSize
        self.actionSize = len(moveList) if hasattr(moveList, '__len__') else 20
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.episode_count = 0
        self.update_target_every = 6
        self._last_logged_epsilon = None
        
        self.model = self.initializeNetwork()
        if resume:
            self.loadModel()
            self.loadStats()
            logger.info(f"Resumed training for {self.name} from existing model")
        else:
            logger.info(f"Starting fresh training for {self.name}")
        
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())
        
        if resume and hasattr(self, "total_timesteps"):
            self.calculateEpsilonFromTimesteps()
            logger.info(f"Resuming with epsilon: {self.epsilon}")
        else:
            self.epsilon = 1.0
            logger.info(f"Starting with epsilon: {self.epsilon}")

    def prepareForNextFight(self):
        """Reset agent memory and episode rewards for a new fight"""
        self.memory = CircularBuffer(DeepQAgent.MAX_DATA_LENGTH)
        self.episode_rewards = []

    def getRandomMove(self, info):
        """Get a random move from the available move list"""
        moveName = random.choice(list(self.moveList))
        frameInputs = self.convertMoveToFrameInputs(moveName, info)
        return moveName.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert a move to frame inputs"""
        frameInputs = self.moveList.getMoveInputs(move)
        frameInputs = self.formatInputsForDirection(move, frameInputs, info)
        return frameInputs

    def formatInputsForDirection(self, move, frameInputs, info):
        """Format move inputs based on player and enemy positions"""
        if not self.moveList.isDirectionalMove(move):
            return frameInputs
        if "x_position" not in info:
            info["x_position"] = 100
        if "enemy_x_position" not in info:
            info["enemy_x_position"] = 200
            
        if info["x_position"] < info["enemy_x_position"]:
            return (frameInputs[0] if isinstance(frameInputs, list) and len(frameInputs) > 0 else frameInputs)
        else:
            return (frameInputs[1] if isinstance(frameInputs, list) and len(frameInputs) > 1 else frameInputs)

    def recordStep(self, step):
        """
        Record a step with balanced reward shaping
        
        Args:
            step: List containing state, action, reward, next_state and done information
        """
        modified_step = list(step)
        
        current_state = step[DeepQAgent.STATE_INDEX]
        next_state = step[DeepQAgent.NEXT_STATE_INDEX]
        reward = step[DeepQAgent.REWARD_INDEX]
        
        current_player_health = current_state.get("health", 100) if current_state else 100
        current_enemy_health = current_state.get("enemy_health", 100) if current_state else 100
        next_player_health = next_state.get("health", 100) if next_state else 100
        next_enemy_health = next_state.get("enemy_health", 100) if next_state else 100
        
        damage_dealt = max(0, current_enemy_health - next_enemy_health)
        damage_taken = max(0, current_player_health - next_player_health)
        
        player_x = current_state.get("x_position", 0) if current_state else 0
        enemy_x = current_state.get("enemy_x_position", 0) if current_state else 0
        distance = abs(player_x - enemy_x)
        
        combo_count = current_state.get("combo_count", 0) if current_state else 0
        
        # FIX #2: Balance reward scaling to avoid extreme values
        damage_reward = damage_dealt * 0.05  # Reduced from 0.2
        damage_penalty = damage_taken * -0.05  # Reduced from -0.2
        
        # Score-based reward
        current_score = current_state.get('score', 0) if current_state else 0
        next_score = next_state.get('score', 0) if next_state else 0
        score_increase = max(0, next_score - current_score)
        
        modified_reward = reward + damage_reward + damage_penalty + (score_increase * 0.005)  # Reduced from 0.01
        
        # FIX #2: Scale win/loss rewards to be more balanced
        if next_enemy_health <= 0:
            modified_reward += 20 + (next_player_health / 100 * 10)  # Reduced from 150 + (health / 100 * 50)
        elif next_player_health <= 0:
            modified_reward -= 15  # Reduced from -75
        
        combo_bonus = min(combo_count, 10) * 0.5  # Cap and reduce combo bonus
        modified_reward += combo_bonus
        
        if distance < 50:
            modified_reward += 0.05 * (50 - distance) / 50  # Normalized distance bonus
        
        modified_step[DeepQAgent.REWARD_INDEX] = modified_reward
        
        # Store the experience in memory
        self.memory.append(modified_step)
        self.total_timesteps += 1
        self.episode_rewards.append(modified_reward)

    def updateEpisodeMetrics(self):
        """Update metrics at the end of an episode"""
        self.episodes_completed += 1
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.avg_reward_history.append(avg_reward)
            self.episode_rewards = []

    def reviewFight(self):
        """
        Review and learn from the previous fight (episode)
        Updates the target network, trains the model, and updates metrics
        """
        self.episode_count += 1
        
        # Determine outcome using matches_won and enemy_matches_won
        if self.memory.size > 0:
            experiences = self.memory.get_all()
            last_experience = experiences[-1]
            current_state = last_experience[DeepQAgent.STATE_INDEX]
            next_state = last_experience[DeepQAgent.NEXT_STATE_INDEX]
            
            current_matches_won = current_state.get('matches_won', 0) if current_state else 0
            next_matches_won = next_state.get('matches_won', 0) if next_state else 0
            current_enemy_matches_won = current_state.get('enemy_matches_won', 0) if current_state else 0
            next_enemy_matches_won = next_state.get('enemy_matches_won', 0) if next_state else 0
            
            if next_matches_won > current_matches_won:
                self.episode_outcome = 1  # Win
            elif next_enemy_matches_won > current_enemy_matches_won:
                self.episode_outcome = 0  # Loss
            else:
                self.episode_outcome = 0  # Draw or no change
        else:
            self.episode_outcome = 0
        
        self.episode_outcomes.append(self.episode_outcome)
        
        if self.episode_count % self.update_target_every == 0:
            if hasattr(self, "target_model"):
                self.target_model.set_weights(self.model.get_weights())
                logger.info(f"Target network updated at episode {self.episode_count}")
        
        try:
            training_data, indices = self.prepareMemoryForTraining(self.memory)
            if len(training_data) > 0:
                original_lr = self.model.optimizer.learning_rate.numpy()
                temp_lr = original_lr * 0.5
                self.model.optimizer.learning_rate.assign(temp_lr)
                self.model, td_errors = self.trainNetwork(training_data, self.model, indices)
                self.model.optimizer.learning_rate.assign(original_lr)
                
                # FIX #4: Update priorities based on TD errors
                if td_errors is not None and indices:
                    self.memory.update_priorities(indices, td_errors)
        except Exception as e:
            logger.error(f"Error training network: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
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
        """Load model weights from file if available"""
        models_dir = self.DEFAULT_MODELS_DIR_PATH
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
        """Save model weights to file"""
        models_dir = self.DEFAULT_MODELS_DIR_PATH
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{self.getModelName()}.weights.h5")
        try:
            self.model.save_weights(model_path)
            logger.info(f"✓ Model weights saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def saveStats(self):
        """Save agent statistics and memory to file"""
        stats_dir = self.DEFAULT_STATS_DIR_PATH
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
            "episode_outcomes": self.episode_outcomes,
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
        """Load agent statistics and memory from file"""
        stats_dir = self.DEFAULT_STATS_DIR_PATH
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
                    self.episode_count = stats.get("episode_count", 0)
                    self.episode_outcomes = stats.get("episode_outcomes", [])
                    if hasattr(self, "calculateEpsilonFromTimesteps"):
                        self.calculateEpsilonFromTimesteps()
                    logger.info(f"✓ Loaded stats from {stats_path}")
            except Exception as e:
                logger.error(f"Error loading stats: {e}")
        
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "rb") as file:
                    memory_data = pickle.load(file)
                    self.memory = CircularBuffer(DeepQAgent.MAX_DATA_LENGTH)
                    for experience in memory_data:
                        self.memory.append(experience)
                    logger.info(f"✓ Loaded memory with {len(memory_data)} experiences")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

    def printTrainingProgress(self):
        """Print current training progress"""
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
        if self.episode_outcomes:
            win_rate = sum(self.episode_outcomes[-20:]) / min(len(self.episode_outcomes), 20)
            logger.info(f"Recent win rate: {win_rate:.4f}")
        logger.info("===========================\n")

    def printFinalStats(self):
        """Print final training statistics"""
        elapsed_time = time.time() - self.training_start_time
        logger.info("\n======= TRAINING SUMMARY =======")
        logger.info(f"Total training timesteps: {self.total_timesteps}")
        logger.info(f"Total episodes completed: {self.episodes_completed}")
        logger.info(f"Total training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            logger.info(f"Final average reward: {self.avg_reward_history[-1]:.4f}")
        if self.avg_loss_history:
            logger.info(f"Final average loss: {self.avg_loss_history[-1]:.6f}")
        if self.episode_outcomes:
            win_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
            logger.info(f"Overall win rate: {win_rate:.4f}")
        logger.info("=================================\n")

    def getModelName(self):
        """Get the name of the model for file operations"""
        return self.name + "Model"

    def getLogsName(self):
        """Get the name of the logs for file operations"""
        return self.name + "Logs"

    def getMove(self, obs, info):
        """
        Get the next move using an epsilon-greedy policy.

        Args:
            obs: Current observation
            info: Additional information about the game state

        Returns:
            tuple: (move, frameInputs) where move is the selected move index and frameInputs are the frame inputs for the move
        """
        # Ensure required fields are present in info
        for field in ["matches_won", "enemy_matches_won", "score"]:
            if field not in info:
                info[field] = 0
                logger.warning(f"{field} not found in info, defaulting to 0")

        # Epsilon-greedy policy: random move if epsilon is met
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs

        # Otherwise, predict move using the model
        try:
            state_data = self.prepareNetworkInputs(info)

            # Select device (GPU if available, else CPU)
            device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"

            # Perform prediction using the existing model
            with tf.device(device):
                q_values = self.model.predict(state_data, verbose=0)

                # Ensure q_values has the expected shape
                if q_values.ndim == 1:
                    q_values = np.reshape(q_values, (1, -1))

                if q_values.shape[1] != self.actionSize:
                    logger.warning(f"Model output shape mismatch: expected {self.actionSize}, got {q_values.shape[1]}")
                    return self.getRandomMove(info)

                # Select the move with the highest Q-value
                move = np.argmax(q_values[0])

                # Ensure move is valid within moveList
                if move >= len(self.moveList):
                    move = move % len(self.moveList)

                move_enum = list(self.moveList)[move]
                frameInputs = self.convertMoveToFrameInputs(move_enum, info)

                return move, frameInputs

        except Exception as e:
            logger.error(f"Error in getMove: {e}")
            return self.getRandomMove(info)

    def initializeNetwork(self):
        """
        Initialize Dueling DQN network architecture
        
        Returns:
            Compiled Keras model
        """
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
            q_values = Lambda(
                lambda a: a[0] + (a[1] - tf.math.reduce_mean(a[1], axis=1, keepdims=True)),
                output_shape=(self.actionSize,)
            )([value, advantage])
            
            model = Model(inputs=input_layer, outputs=q_values)
            optimizer = Adam(learning_rate=self.learningRate, clipnorm=1.0)
            model.compile(loss=_huber_loss, optimizer=optimizer)
            
            params = sum([np.prod(w.shape) for w in model.trainable_weights])
            logger.info(f"Initialized Dueling DQN with residual connections (~{params:,} parameters)")
            
            return model

    def prepareMemoryForTraining(self, memory):
        """
        Prepare memory for training with prioritized experience replay
        
        Args:
            memory: CircularBuffer object containing experiences
            
        Returns:
            List of experiences and indices for training
        """
        # FIX #4: Use proper prioritized experience replay
        experiences, indices = memory.sample(2048, use_priorities=True)
        if not experiences:
            return [], []
        
        formatted_data = []
        for step in experiences:
            if len(step) < 7:  # Make sure we have all required components
                continue
                
            state = step[DeepQAgent.STATE_INDEX]
            next_state = step[DeepQAgent.NEXT_STATE_INDEX]
            action = step[DeepQAgent.ACTION_INDEX]
            reward = step[DeepQAgent.REWARD_INDEX]
            done = step[DeepQAgent.DONE_INDEX]
            
            processed_state = self.prepareNetworkInputs(state)
            processed_next_state = self.prepareNetworkInputs(next_state)
            
            formatted_data.append([processed_state, action, reward, done, processed_next_state])
        
        # Select a batch size based on available data
        batch_size = min(64, len(formatted_data))
        selected_indices = random.sample(range(len(formatted_data)), batch_size) if batch_size > 0 else []
        selected_data = [formatted_data[i] for i in selected_indices]
        selected_original_indices = [indices[i] for i in selected_indices]
        
        return selected_data, selected_original_indices

    def prepareNetworkInputs(self, step):
        """
        Convert game state information to network input features
        
        Args:
            step: Dictionary containing game state information
            
        Returns:
            Numpy array with processed features
        """
        if step is None:
            return np.zeros((1, self.stateSize), dtype=np.float32)
            
        feature_vector = []
        
        # FIX #1: Ensure all features are properly included without truncation
        # Health-related features (5)
        player_health = step.get("health", 100)
        enemy_health = step.get("enemy_health", 100)
        player_health_pct = player_health / 100.0
        enemy_health_pct = enemy_health / 100.0
        health_advantage = player_health_pct - enemy_health_pct
        feature_vector.extend([player_health, enemy_health, player_health_pct, enemy_health_pct, health_advantage])
        
        # Position-related features (7)
        player_x = step.get("x_position", 0)
        player_y = step.get("y_position", 0)
        enemy_x = step.get("enemy_x_position", 0)
        enemy_y = step.get("enemy_y_position", 0)
        x_distance = abs(player_x - enemy_x)
        y_distance = abs(player_y - enemy_y)
        euclidean_distance = np.sqrt(x_distance**2 + y_distance**2)
        feature_vector.extend([player_x, player_y, enemy_x, enemy_y, x_distance, y_distance, euclidean_distance])
        
        # Direction features (2)
        facing_right = 1 if player_x < enemy_x else 0
        enemy_above = 1 if player_y > enemy_y else 0
        feature_vector.extend([facing_right, enemy_above])
        
        # Match statistics (3)
        matches_won = step.get("matches_won", 0)
        enemy_matches_won = step.get("enemy_matches_won", 0)
        score = step.get("score", 0.0)
        feature_vector.extend([matches_won, enemy_matches_won, score / 1000.0])
        
        # Enemy state one-hot encoding (9)
        enemy_status = step.get("enemy_status", 512)
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(enemy_status, 0)
        oneHotEnemyState[state_index] = 1
        feature_vector.extend(oneHotEnemyState)
        
        # Enemy character one-hot encoding (8)
        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector.extend(oneHotEnemyChar)
        
        # Player state one-hot encoding (9)
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(step.get("status", 512), 0)
        oneHotPlayerState[state_index] = 1
        feature_vector.extend(oneHotPlayerState)
        
        # Ensure the feature vector is exactly the expected size
        if len(feature_vector) < self.stateSize:
            feature_vector.extend([0] * (self.stateSize - len(feature_vector)))
        elif len(feature_vector) > self.stateSize:
            feature_vector = feature_vector[:self.stateSize]
        
        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    def trainNetwork(self, data, model, indices):
        """
        Train the network using experience data
        
        Args:
            data: List of experience tuples
            model: The model to train
            indices: Indices for prioritized replay updates
            
        Returns:
            Updated model and TD errors for priority updates
        """
        if len(data) == 0:
            logger.warning("No data available for training. Skipping.")
            return model, None
        
        batch_size = min(64, len(data))
        if batch_size == 0:
            return model, None
        
        states_batch = []
        targets_batch = []
        td_errors = []
        
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        
        with tf.device(device):
            try:
                for experience in data:
                    if len(experience) < 5:
                        continue
                        
                    state = experience[0]
                    action = experience[1]
                    reward = experience[2]
                    done = experience[3]
                    next_state = experience[4]
                    
                    if action >= self.actionSize:
                        action = action % self.actionSize
                    
                    # Get current Q values
                    current_q = model.predict(state, verbose=0)[0]
                    old_val = current_q[action]
                    
                    # Calculate target Q value using Double DQN
                    if done:
                        target = reward
                    else:
                        # Double DQN: select action using online network
                        next_action = np.argmax(model.predict(next_state, verbose=0)[0])
                        # Evaluate action using target network
                        next_q = self.target_model.predict(next_state, verbose=0)[0][next_action]
                        target = reward + self.gamma * next_q
                    
                    # Update Q value for the selected action
                    current_q[action] = target
                    
                    # Calculate TD error for prioritized replay
                    td_error = abs(old_val - target)
                    td_errors.append(td_error)
                    
                    states_batch.append(state[0])
                    targets_batch.append(current_q)
                
                if not states_batch:
                    logger.warning("No valid experiences in batch. Skipping training.")
                    return model, None
                
                states_array = np.array(states_batch)
                targets_array = np.array(targets_batch)
                
                # Train the model with the batch
                history = model.fit(
                    states_array, 
                    targets_array,
                    batch_size=min(32, len(states_array)),
                    epochs=1,
                    verbose=0
                )
                
                if hasattr(self, 'lossHistory') and 'loss' in history.history:
                    loss_value = history.history['loss'][0]
                    self.lossHistory.on_epoch_end(0, {'loss': float(loss_value)})
                
            except Exception as e:
                logger.error(f"Error in trainNetwork: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return model, None
                
        return model, td_errors

    def calculateEpsilonFromTimesteps(self):
        """
        Calculate epsilon value based on timesteps with consistent decay schedule
        """
        START_EPSILON = 1.0
        TIMESTEPS_TO_MIN_EPSILON = 1500000
        
        decay_per_step = (START_EPSILON - DeepQAgent.EPSILON_MIN) / TIMESTEPS_TO_MIN_EPSILON
        self.epsilon = max(DeepQAgent.EPSILON_MIN, START_EPSILON - (decay_per_step * self.total_timesteps))
        
        if self.episodes_completed > 20:
            win_rate = sum(self.episode_outcomes[-20:]) / min(len(self.episode_outcomes), 20) if self.episode_outcomes else 0
            if win_rate < 0.15:
                self.epsilon = min(0.8, self.epsilon + 0.15)
                logger.info(f"Performance boost: epsilon increased to {self.epsilon} due to low win rate")
        
        if hasattr(self, '_last_logged_epsilon') and self._last_logged_epsilon is not None:
            if abs(self._last_logged_epsilon - self.epsilon) > 0.05:
                logger.info(f"Epsilon updated: {self._last_logged_epsilon:.4f} -> {self.epsilon:.4f}")
                self._last_logged_epsilon = self.epsilon
        else:
            self._last_logged_epsilon = self.epsilon

    def updateEpsilon(self):
        """Update epsilon based on current timesteps"""
        self.calculateEpsilonFromTimesteps()

if __name__ == "__main__":
    agent = DeepQAgent(stateSize=45, resume=True)
    logger.info(f"Agent initialized with epsilon {agent.epsilon}")
    
    for episode in range(10):
        agent.prepareForNextFight()
        
        for step in range(100):
            state = {
                "health": 100 - step // 2,
                "enemy_health": 100 - step,
                "x_position": 100 + step,
                "y_position": 50,
                "enemy_x_position": 200 - step // 2,
                "enemy_y_position": 50,
                "status": 512,
                "enemy_status": 514,
                "combo_count": step // 20,
                "matches_won": episode // 2,
                "enemy_matches_won": episode // 3,
                "score": 1000 + step * 10,
            }
            
            move, inputs = agent.getMove(None, state)
            
            next_state = {
                "health": max(0, state["health"] - (1 if random.random() < 0.3 else 0)),
                "enemy_health": max(0, state["enemy_health"] - (5 if random.random() < 0.4 else 0)),
                "x_position": state["x_position"] + random.randint(-10, 10),
                "y_position": state["y_position"] + random.randint(-5, 5),
                "enemy_x_position": state["enemy_x_position"] + random.randint(-10, 10),
                "enemy_y_position": state["enemy_y_position"] + random.randint(-5, 5),
                "status": state["status"],
                "enemy_status": state["enemy_status"],
                "combo_count": state["combo_count"] + (1 if random.random() < 0.1 else 0),
                "matches_won": state["matches_won"] + (1 if random.random() < 0.2 else 0),
                "enemy_matches_won": state["enemy_matches_won"] + (1 if random.random() < 0.15 else 0),
                "score": state["score"] + random.randint(0, 50),
                "match_duration": step,
            }
            
            reward = 1.0 if next_state["enemy_health"] < state["enemy_health"] else -0.5
            done = next_state["health"] <= 0 or next_state["enemy_health"] <= 0 or next_state["matches_won"] > state["matches_won"] or next_state["enemy_matches_won"] > state["enemy_matches_won"]
            
            agent.recordStep([None, state, move, reward, None, next_state, done])
            
            if done:
                break
        
        agent.reviewFight()
        logger.info(f"Episode {episode+1} completed. Epsilon: {agent.epsilon:.4f}")
    
    agent.saveModel()
    agent.saveStats()
    agent.printFinalStats()