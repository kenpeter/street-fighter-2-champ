import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Lambda, Input
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
            # Sample with replacement based on weights
            indices = random.choices(range(self.size), weights=weights[:self.size], k=min(batch_size, self.size))
        else:
            # Random sampling without replacement
            indices = random.sample(range(self.size), min(batch_size, self.size))
        return [self.buffer[i] for i in indices], indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for specific indices"""
        for idx, priority in zip(indices, priorities):
            if self.buffer[idx] is not None:
                self.buffer[idx][-1] = priority  # Update priority value
    
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
    PRIORITY_INDEX = 7  # Added index for priority
    MAX_DATA_LENGTH = 200000  # Requirement 1: Increased from 50000 to 200000 (4x)
    DEFAULT_MODELS_DIR_PATH = "./models"
    DEFAULT_LOGS_DIR_PATH = "./logs"
    DEFAULT_STATS_DIR_PATH = "./stats"

    def __init__(self, resume=False, name=None, moveList=Moves):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
            
        # Get current directory for absolute paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Update class variables with absolute paths
        Agent.DEFAULT_MODELS_DIR_PATH = os.path.join(self.current_dir, "models")
        Agent.DEFAULT_LOGS_DIR_PATH = os.path.join(self.current_dir, "logs")
        Agent.DEFAULT_STATS_DIR_PATH = os.path.join(self.current_dir, "stats")
        
        # Create necessary directories with absolute paths
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

        # Learning rate decay tracking
        self.lr_step_size = 50000  # CHANGE: Apply decay every 50,000 timesteps (was 10,000)
        self.last_lr_update = 0  # Timestep of last update
        
        # Setup the model if this is a proper agent subclass
        if self.__class__.__name__ != "Agent":
            self.model = self.initializeNetwork()
            # Resume training if requested
            if resume:
                self.loadModel()
                self.loadStats()
                logger.info(f"Resumed training for {self.name} from existing model")
            else:
                logger.info(f"Starting fresh training for {self.name}")

    def prepareForNextFight(self):
        # Use circular buffer instead of deque
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
        # CHANGE: Add win/loss reward logic
        modified_step = list(step)
        reward = modified_step[Agent.REWARD_INDEX]
        
        # Add large win bonus or loss penalty
        enemy_health = step[Agent.STATE_INDEX].get("enemy_health", 100)
        player_health = step[Agent.STATE_INDEX].get("health", 100)
        
        if enemy_health <= 0:
            reward += 100  # Large win bonus
        if player_health <= 0:
            reward -= 50   # Penalize losing
        
        # Update the reward in the step
        modified_step[Agent.REWARD_INDEX] = reward
        
        # Use a default priority initially based on reward magnitude
        priority = abs(reward) + 0.01  # Small constant to avoid zero priority
        
        # Append priority
        modified_step_with_priority = modified_step + [priority]
        
        # Add to memory buffer
        self.memory.append(modified_step_with_priority)
        
        # Update counters
        self.total_timesteps += 1
        self.episode_rewards.append(reward)  # Use modified reward

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
        # Use absolute path for model loading
        models_dir = os.path.join(self.current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Debug output
        logger.info(f"Looking for model files in: {models_dir}")
        
        # Try loading with different extensions
        model_path = os.path.join(models_dir, f"{self.getModelName()}")
        load_successful = False
        for ext in [".weights.h5", ".h5", ".keras"]:
            full_path = model_path + ext
            if os.path.exists(full_path):
                logger.info(f"Found model file: {full_path}")
                try:
                    self.model.load_weights(full_path)
                    load_successful = True
                    logger.info(f"✓ Model loaded successfully from {full_path}!")
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {full_path}: {e}")
        
        # Sync target network if we have one and the model loaded successfully
        if load_successful and hasattr(self, "target_model"):
            self.target_model.set_weights(self.model.get_weights())
            logger.info("Target network synchronized with loaded model weights")
        
        if not load_successful:
            logger.warning(f"No valid model file found, will use a new model.")

    def saveModel(self):
        try:
            # Use consistent absolute path for model saving
            models_dir = os.path.join(self.current_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Save weights with consistent naming
            model_path = os.path.join(models_dir, f"{self.getModelName()}.weights.h5")
            
            logger.info(f"Saving model to: {model_path}")
            
            # Save with error handling
            try:
                self.model.save_weights(model_path)
                logger.info(f"✓ Model weights saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model weights: {e}")
                # Try alternate save method
                alt_path = os.path.join(models_dir, f"{self.getModelName()}.keras")
                try:
                    self.model.save(alt_path)
                    logger.info(f"✓ Model saved using alternative method to {alt_path}")
                except Exception as e2:
                    logger.error(f"Alternative save also failed: {e2}")
            
            # Save logs
            logs_dir = os.path.join(self.current_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logs_path = os.path.join(logs_dir, f"{self.getLogsName()}")
            
            logger.info(f"Saving logs to: {logs_path}")
            with open(logs_path, "a+") as file:
                if (
                    hasattr(self, "lossHistory")
                    and hasattr(self.lossHistory, "losses")
                    and len(self.lossHistory.losses) > 0
                ):
                    avg_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
                    file.write(f"{avg_loss}\n")
                    logger.info(f"✓ Logs saved to {logs_path}")
        except Exception as e:
            logger.error(f"Critical error in saveModel: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def saveStats(self):
        try:
            # Use consistent absolute path for stats
            stats_dir = os.path.join(self.current_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            
            # Save with consistent naming
            stats_path = os.path.join(stats_dir, f"{self.name}_stats.json")
            memory_path = os.path.join(stats_dir, f"{self.name}_memory.pkl")
            
            logger.info(f"Saving stats to: {stats_path}")
            logger.info(f"Saving memory to: {memory_path}")
            
            # Gather all relevant state
            stats = {
                "total_timesteps": self.total_timesteps,
                "episodes_completed": self.episodes_completed,
                "avg_reward_history": self.avg_reward_history,
                "avg_loss_history": self.avg_loss_history,
                "last_lr_update": getattr(self, "last_lr_update", 0),
                "episode_count": getattr(self, "episode_count", 0),  # For target network updates
            }
            
            # Save stats JSON
            try:
                with open(stats_path, "w") as file:
                    json.dump(stats, file, indent=4)
                logger.info(f"✓ Stats saved to {stats_path}")
            except Exception as e:
                logger.error(f"Error saving stats to {stats_path}: {e}")
                # Try alternate location
                alt_path = os.path.join(self.current_dir, f"{self.name}_stats.json")
                try:
                    with open(alt_path, "w") as file:
                        json.dump(stats, file, indent=4)
                    logger.info(f"✓ Stats saved to alternate location: {alt_path}")
                except Exception as e2:
                    logger.error(f"Alternative stats save also failed: {e2}")

            # Save memory buffer
            try:
                with open(memory_path, "wb") as file:
                    # Get all valid entries from the circular buffer
                    memory_data = self.memory.get_all()
                    pickle.dump(memory_data, file)
                logger.info(f"✓ Memory buffer saved with {len(memory_data)} experiences")
            except Exception as e:
                logger.error(f"Error saving memory buffer: {e}")
        except Exception as e:
            logger.error(f"Critical error in saveStats: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def loadStats(self):
        # Use consistent absolute path for stats
        stats_dir = os.path.join(self.current_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        logger.info(f"Looking for stats files in: {stats_dir}")
        
        stats_path = os.path.join(stats_dir, f"{self.name}_stats.json")
        memory_path = os.path.join(stats_dir, f"{self.name}_memory.pkl")
        
        self.loaded_stats = False
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as file:
                    stats = json.load(file)
                    self.total_timesteps = stats.get("total_timesteps", 0)
                    self.episodes_completed = stats.get("episodes_completed", 0)
                    self.avg_reward_history = stats.get("avg_reward_history", [])
                    self.avg_loss_history = stats.get("avg_loss_history", [])
                    self.last_lr_update = stats.get("last_lr_update", 0)
                    
                    # Set episode count for target network updates
                    if hasattr(self, "episode_count"):
                        self.episode_count = stats.get("episode_count", 0)
                    
                    # Calculate epsilon based on total timesteps
                    if hasattr(self, "calculateEpsilonFromTimesteps"):
                        self.calculateEpsilonFromTimesteps()
                    
                    self.loaded_stats = True
                    logger.info(f"✓ Loaded training stats from {stats_path}")
                    logger.info(f"  - Timesteps: {self.total_timesteps}")
                    logger.info(f"  - Episodes: {self.episodes_completed}")
                    if hasattr(self, "epsilon"):
                        logger.info(f"  - Calculated epsilon: {self.epsilon}")
                    if hasattr(self, "episode_count"):
                        logger.info(f"  - Episode count (for target network): {self.episode_count}")
            except Exception as e:
                logger.error(f"Error loading stats from {stats_path}: {e}")
                logger.warning("Starting with fresh training statistics.")
        else:
            logger.warning(f"No stats file found at {stats_path}. Starting fresh.")

        if os.path.exists(memory_path):
            try:
                with open(memory_path, "rb") as file:
                    memory_data = pickle.load(file)
                    # Initialize a new circular buffer
                    self.memory = CircularBuffer(Agent.MAX_DATA_LENGTH)
                    # Add each experience to the buffer
                    for experience in memory_data:
                        self.memory.append(experience)
                    logger.info(f"✓ Loaded memory buffer with {len(memory_data)} experiences")
            except Exception as e:
                logger.error(f"Error loading memory buffer: {e}")
                self.memory = CircularBuffer(Agent.MAX_DATA_LENGTH)
        else:
            self.memory = CircularBuffer(Agent.MAX_DATA_LENGTH)
            logger.warning("No previous memory buffer found. Starting with empty buffer.")

    def printTrainingProgress(self):
        elapsed_time = time.time() - self.training_start_time
        logger.info("\n==== Training Progress ====")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        logger.info(f"Episodes completed: {self.episodes_completed}")
        logger.info(f"Training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            logger.info(f"Recent average reward: {self.avg_reward_history[-1]:.4f}")
            if len(self.avg_reward_history) >= 2:
                reward_change = (
                    self.avg_reward_history[-1] - self.avg_reward_history[-2]
                )
                logger.info(f"Reward change: {reward_change:+.4f}")
        if (
            hasattr(self, "lossHistory")
            and hasattr(self.lossHistory, "losses")
            and len(self.lossHistory.losses) > 0
        ):
            recent_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
            logger.info(f"Recent loss: {recent_loss:.6f}")
            if self.avg_loss_history and len(self.avg_loss_history) >= 2:
                loss_change = self.avg_loss_history[-1] - self.avg_loss_history[-2]
                logger.info(f"Loss change: {loss_change:+.6f}")
                if abs(loss_change) < 0.0001 and self.episodes_completed > 5:
                    logger.warning(
                        "WARNING: Training may be stuck in a local minimum - loss is not changing significantly"
                    )
                elif loss_change < 0:
                    logger.info("Learning progress: Positive (loss is decreasing)")
                else:
                    logger.warning(
                        "Learning progress: Negative or stalled (loss is not decreasing)"
                    )
        logger.info("===========================\n")

    def printFinalStats(self):
        elapsed_time = time.time() - self.training_start_time
        logger.info("\n======= TRAINING SUMMARY =======")
        logger.info(f"Total training timesteps: {self.total_timesteps}")
        logger.info(f"Total episodes completed: {self.episodes_completed}")
        logger.info(f"Total training time: {elapsed_time:.2f} seconds")
        if self.avg_reward_history:
            logger.info(f"Final average reward: {self.avg_reward_history[-1]:.4f}")
            if len(self.avg_reward_history) > 1:
                first_rewards = sum(self.avg_reward_history[:3]) / min(3, len(self.avg_reward_history))
                last_rewards = sum(self.avg_reward_history[-3:]) / min(3, len(self.avg_reward_history))
                reward_improvement = last_rewards - first_rewards
                logger.info(f"Reward improvement: {reward_improvement:+.4f}")
        if self.avg_loss_history:
            logger.info(f"Final average loss: {self.avg_loss_history[-1]:.6f}")
            if len(self.avg_loss_history) > 1:
                first_losses = sum(self.avg_loss_history[:3]) / min(3, len(self.avg_loss_history))
                last_losses = sum(self.avg_loss_history[-3:]) / min(3, len(self.avg_loss_history))
                loss_improvement = first_losses - last_losses
                logger.info(f"Loss improvement: {loss_improvement:+.6f}")
                learning_status = (
                    "POSITIVE - Agent is learning effectively" if loss_improvement > 0 else
                    "NEUTRAL - Small improvements in learning" if loss_improvement > -0.001 else
                    "NEGATIVE - Agent may be stuck in suboptimal policy"
                )
                logger.info(f"Learning status: {learning_status}")
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
        raise NotImplementedError("Implement this in the inherited agent")

    def trainNetwork(self, data, model):
        raise NotImplementedError("Implement this in the inherited agent")

class DeepQAgent(Agent):
    # CHANGE: Lower min epsilon from 0.1 to 0.05 for better exploration
    EPSILON_MIN = 0.05  
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
        resume=False,
        name=None,
        moveList=Moves,
    ):
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        
        # Extra safety check for moveList
        if hasattr(moveList, '__len__'):
            self.actionSize = len(moveList)
            logger.info(f"Action space size from moveList: {self.actionSize}")
        else:
            # Fallback if moveList doesn't have a length
            logger.warning("MoveList doesn't have a length. Using default action size.")
            self.actionSize = 20  # Use a reasonable default
        
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.total_timesteps = 0
        self.episode_count = 0  # For target network updates
        self.update_target_every = 5  # Requirement 2: Update target network every 5 episodes
        self.lr_decay = 0.995  # Requirement 4: Learning rate decay factor
        
        super(DeepQAgent, self).__init__(resume=resume, name=name, moveList=moveList)
        
        # Print action space information for debugging
        if hasattr(moveList, '__len__'):
            logger.info(f"MoveList contains {len(moveList)} actions")
            for i, move in enumerate(moveList):
                logger.info(f"  Action {i}: {move}")
                
        # Requirement 2: Initialize target network
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())
        
        # Verify output layer size matches action space
        try:
            # Get output shape directly from the model
            output_shape = self.model.output_shape
        except AttributeError:
            # Fallback if output_shape isn't available
            logger.warning("Could not access model.output_shape directly. Using alternate method.")
            output_shape = (None, self.actionSize)  # Assume shape matches action size
            
        if output_shape[-1] != self.actionSize:
            logger.error(f"Model output size ({output_shape[-1]}) doesn't match action space ({self.actionSize})")
            logger.info("Rebuilding model with correct action space size")
            self.model = self.initializeNetwork()
            self.target_model = self.initializeNetwork()
            self.target_model.set_weights(self.model.get_weights())


        # Handle epsilon (exploration rate)
        if resume and hasattr(self, "total_timesteps"):
            # When resuming, calculate epsilon based on total timesteps
            self.calculateEpsilonFromTimesteps()
            logger.info(f"Resuming with calculated epsilon: {self.epsilon} based on {self.total_timesteps} timesteps")
        else:
            # Start fresh with full exploration
            self.epsilon = 1.0
            logger.info(f"Starting with new epsilon: {self.epsilon}")
            
        # Verify model input shape matches state size
        try:
            # Get input shape directly from the model
            input_shape = self.model.input_shape
        except AttributeError:
            # Fallback if input_shape isn't available directly
            logger.warning("Could not access model.input_shape directly. Using alternate method.")
            try:
                # Try to get input_shape from the first layer's output shape
                input_shape = self.model.layers[0].output_shape
            except (AttributeError, IndexError):
                # If that fails, assume it matches state size
                input_shape = (None, self.stateSize)
                
        if input_shape[1] != self.stateSize:
            logger.warning(f"Model input shape ({input_shape[1]}) doesn't match state size ({self.stateSize})")
            # We'll handle this dynamically in prepareNetworkInputs



    def calculateEpsilonFromTimesteps(self):
        """Calculate epsilon based on the total training timesteps"""
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
        """Update epsilon based on current training progress"""
        old_epsilon = self.epsilon
        self.calculateEpsilonFromTimesteps()
        if abs(old_epsilon - self.epsilon) > 0.01:  # Only log if changed significantly
            logger.info(f"Epsilon updated: {old_epsilon:.4f} -> {self.epsilon:.4f} at timestep {self.total_timesteps}")


    # Modify the getMove method to handle the shape mismatch
    def getMove(self, obs, info):
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs
        else:
            stateData = self.prepareNetworkInputs(info)
            device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
            
            try:
                with tf.device(device):
                    # Get predictions
                    predictions = self.model.predict(stateData, verbose=0)
                    
                    # Check if predictions shape needs reshaping
                    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                        logger.info("Detected scalar output, reshaping to match action space")
                        # Create a uniform distribution as a fallback
                        predictedRewards = np.ones((self.actionSize,)) / self.actionSize
                    elif len(predictions.shape) == 2 and predictions.shape[1] != self.actionSize:
                        logger.warning(f"Model output shape {predictions.shape} doesn't match action space ({self.actionSize})")
                        logger.warning("Using random action distribution")
                        # Create a uniform distribution as a fallback
                        predictedRewards = np.ones((self.actionSize,)) / self.actionSize
                    else:
                        # Use predictions as is
                        predictedRewards = predictions[0]
                    
                    # Get best action
                    move = np.argmax(predictedRewards)
                    
                    # Safety check: if move is out of range
                    if move >= len(self.moveList):
                        logger.warning(f"Predicted move {move} is out of range for moveList size {len(self.moveList)}")
                        logger.warning("Clamping to valid range")
                        move = move % len(self.moveList)
                    
                    frameInputs = self.convertMoveToFrameInputs(list(self.moveList)[move], info)
                    return move, frameInputs
                    
            except Exception as e:
                logger.error(f"Error in getMove: {e}")
                logger.error("Falling back to random move due to prediction error")
                move, frameInputs = self.getRandomMove(info)
                return move, frameInputs
        
    def initializeNetwork(self):
        # Simplified Dueling DQN using pure Functional API
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        with tf.device(device):
            # Input layer
            input_layer = Input(shape=(self.stateSize,))
            
            # Shared network layers - keep it simple
            x = Dense(64, activation='relu')(input_layer)
            x = Dense(128, activation='relu')(x)
            
            # Value Stream - estimates state value
            value_stream = Dense(32, activation='relu')(x)
            value = Dense(1)(value_stream)
            
            # Advantage Stream - estimates advantage of each action
            advantage_stream = Dense(32, activation='relu')(x)
            advantage = Dense(self.actionSize)(advantage_stream)
            
            # Combine value and advantage streams
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            def combine_streams(inputs):
                value, advantage = inputs
                # Use tf instead of K for tensor operations
                value_expanded = tf.expand_dims(value, axis=1)
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
                return value_expanded + (advantage - advantage_mean)
                
            # Define output shape function
            def output_shape(input_shapes):
                return input_shapes[1]  # Return the shape of advantage
            
            # Combine streams with Lambda layer
            q_values = Lambda(combine_streams, output_shape=output_shape)([value, advantage])
            
            # Create complete model
            model = Model(inputs=input_layer, outputs=q_values)
            
            # Compile model
            optimizer = Adam(learning_rate=self.learningRate)
            model.compile(
                loss=DeepQAgent._huber_loss,
                optimizer=optimizer,
            )
            
            logger.info(f"Successfully initialized simplified Dueling DQN model on {device}")
            model.summary(print_fn=lambda x: logger.info(x))
            
        return model


    def prepareMemoryForTraining(self, memory):
        # Get all experiences from the memory buffer
        experiences = memory.get_all()
        if not experiences:
            return []
            
        # Calculate action counts for rarity bonuses
        action_counts = {}
        for step in experiences:
            action = step[Agent.ACTION_INDEX]
            action_counts[action] = action_counts.get(action, 0) + 1
            
        # Compute priorities for each experience
        data_with_priority = []
        beta = 1.0  # Hyperparameter for action rarity weight
        
        for step in experiences:
            state = self.prepareNetworkInputs(step[Agent.STATE_INDEX])
            action = step[Agent.ACTION_INDEX]
            reward = step[Agent.REWARD_INDEX]
            done = step[Agent.DONE_INDEX]
            next_state = self.prepareNetworkInputs(step[Agent.NEXT_STATE_INDEX])
            
            # Calculate action rarity: inverse frequency
            total_experiences = len(experiences)
            action_frequency = action_counts[action] / total_experiences
            rarity_score = beta * (1.0 - action_frequency)
            
            # Priority combines reward magnitude and rarity
            priority = abs(reward) + rarity_score
            
            # Store along with experience
            data_with_priority.append([state, action, reward, done, next_state, priority])
            
        # Sort by priority and select top 70%
        data_with_priority.sort(key=lambda x: x[5], reverse=True)
        top_70_percent = int(0.7 * len(data_with_priority))
        top_data = data_with_priority[:top_70_percent]
        remaining_data = data_with_priority[top_70_percent:]
        
        # Select the remaining 30% to ensure diversity
        # First, track seen states
        seen_states = set()
        for item in top_data:
            state_hash = hash(tuple(item[0].flatten().tolist()))
            seen_states.add(state_hash)
            
        # From remaining, select experiences with unique states
        selected_remaining = []
        for item in remaining_data:
            state_hash = hash(tuple(item[0].flatten().tolist()))
            if state_hash not in seen_states:
                selected_remaining.append(item)
                seen_states.add(state_hash)
                
            # If we have enough, stop selection
            if len(selected_remaining) >= min(int(0.3 * len(data_with_priority)), len(remaining_data)):
                break
                
        # If we don't have enough unique states, add random ones
        if len(selected_remaining) < min(int(0.3 * len(data_with_priority)), len(remaining_data)):
            needed = min(int(0.3 * len(data_with_priority)), len(remaining_data)) - len(selected_remaining)
            random_extra = random.sample(remaining_data, min(needed, len(remaining_data)))
            selected_remaining.extend(random_extra)
            
        # Combine top 70% and selected remaining 30%
        final_data = top_data + selected_remaining
        return final_data

    def prepareNetworkInputs(self, step):
        feature_vector = []
        # e health
        feature_vector.append(step["enemy_health"])
        # e x
        feature_vector.append(step["enemy_x_position"])
        # e y
        feature_vector.append(step["enemy_y_position"])
        
        # Handle unknown status codes safely
        enemy_status = step.get("enemy_status", 512)
        player_status = step.get("status", 512)
        
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(enemy_status, 0)  # Default to first index if unknown
        oneHotEnemyState[state_index] = 1
        
        feature_vector += oneHotEnemyState
        oneHotEnemyChar = [0] * 8
        oneHotEnemyChar[step["enemy_character"]] = 1
        feature_vector += oneHotEnemyChar
        feature_vector.append(step["health"])
        feature_vector.append(step["x_position"])
        feature_vector.append(step["y_position"])
        
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        state_index = DeepQAgent.stateIndices.get(player_status, 0)  # Default to first index if unknown
        oneHotPlayerState[state_index] = 1
        
        feature_vector += oneHotPlayerState
        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def trainNetwork(self, data, model):
        # If no data to train on, return original model
        if len(data) == 0:
            logger.warning("No data available for training. Skipping.")
            return model
            
        # Sample batch from memory
        minibatch, _ = self.memory.sample(128)  # Smaller batch size for stability
        
        # Prepare data for training
        states = []
        targets = []
        
        # Choose the appropriate device
        device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"
        
        with tf.device(device):
            for step in minibatch:
                # Extract necessary components
                state = step[Agent.STATE_INDEX]
                action = step[Agent.ACTION_INDEX]
                reward = step[Agent.REWARD_INDEX]
                next_state = step[Agent.NEXT_STATE_INDEX]
                done = step[Agent.DONE_INDEX]
                
                # Safety check for action index
                if action >= self.actionSize:
                    action = action % self.actionSize
                
                # Prepare state data for network input
                state_data = self.prepareNetworkInputs(state)
                
                # Get current Q values for this state 
                q_values = model.predict(state_data, verbose=0)[0]
                
                # Handle array shape - ensure q_values is the right shape
                if len(q_values.shape) == 0:  # Scalar output
                    q_values = np.ones(self.actionSize) / self.actionSize
                elif len(q_values.shape) > 1:  # Extra dimensions
                    q_values = q_values.flatten()
                    # If flattened size doesn't match actionSize, create uniform distribution
                    if len(q_values) != self.actionSize:
                        q_values = np.ones(self.actionSize) / self.actionSize
                
                # Create a copy of Q values to update
                target = q_values.copy()
                
                if done:
                    # If terminal state, target is just the reward
                    target[action] = reward
                else:
                    # Prepare next state data
                    next_state_data = self.prepareNetworkInputs(next_state)
                    
                    # Use target network for next state Q values
                    next_q_values = self.target_model.predict(next_state_data, verbose=0)[0]
                    
                    # Handle next_q_values shape
                    if len(next_q_values.shape) == 0:  # Scalar output
                        next_q_values = np.ones(self.actionSize) / self.actionSize
                    elif len(next_q_values.shape) > 1:  # Extra dimensions
                        next_q_values = next_q_values.flatten()
                        # If flattened size doesn't match actionSize, create uniform distribution
                        if len(next_q_values) != self.actionSize:
                            next_q_values = np.ones(self.actionSize) / self.actionSize
                    
                    # Update target using Q-learning formula
                    target[action] = reward + self.gamma * np.max(next_q_values)
                
                # Add to training batch
                states.append(state_data[0])
                targets.append(target)
            
            # Convert to numpy arrays
            states = np.array(states)
            targets = np.array(targets)
            
            # Train model on batch
            model.fit(
                states, 
                targets, 
                batch_size=32,
                epochs=1, 
                verbose=0, 
                callbacks=[self.lossHistory]
            )
            
        return model


    # Replace the reviewFight method with this more robust version:
    def reviewFight(self):
        self.episode_count += 1
        
        # Update the target network only periodically
        if self.episode_count % self.update_target_every == 0:
            self.target_model.set_weights(self.model.get_weights())
            logger.info(f"Target network updated at episode {self.episode_count}")
        
        # Prepare data and train network
        try:
            data = self.prepareMemoryForTraining(self.memory)
            if len(data) > 0:
                self.model = self.trainNetwork(data, self.model)
        except Exception as e:
            logger.error(f"Error training network: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Update metrics safely
        try:
            self.updateEpisodeMetrics()
            if (hasattr(self, "lossHistory") and hasattr(self.lossHistory, "losses") and len(self.lossHistory.losses) > 0):
                avg_loss = sum(self.lossHistory.losses) / len(self.lossHistory.losses)
                self.avg_loss_history.append(avg_loss)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        # Update exploration rate
        if hasattr(self, "updateEpsilon"):
            try:
                self.updateEpsilon()
            except Exception as e:
                logger.error(f"Error updating epsilon: {e}")
        
        # Apply learning rate decay if needed
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
        
        # Save progress
        try:
            self.saveModel()
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        try:
            self.saveStats()
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
        
        try:
            self.printTrainingProgress()
        except Exception as e:
            logger.error(f"Error printing progress: {e}")


# Register custom loss function
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