import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Lambda, Add, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import logging
import json
from enum import Enum
import time

# Define Moves Enum and MovesDict
class Moves(Enum):
    """Enum of the set of possible moves the agent is allowed to perform"""
    Idle = 0
    Right = 1
    DownRight = 2
    Down = 3
    DownLeft = 4
    Left = 5
    UpLeft = 6
    Up = 7
    UpRight = 8
    LightPunch = 9
    MediumPunch = 10
    HeavyPunch = 11
    LightKick = 12
    MediumKick = 13
    HeavyKick = 14
    CrouchLightPunch = 15
    CrouchMediumPunch = 16
    CrouchHeavyPunch = 17
    CrouchLightKick = 18
    CrouchMediumKick = 19
    CrouchHeavyKick = 20
    LeftShoulderThrow = 21
    RightShoulderThrow = 22
    LeftSomersaultThrow = 23
    RightSomersaultThrow = 24
    Fireball = 25
    HurricaneKick = 26
    DragonUppercut = 27

    @staticmethod
    def getMoveInputs(moveName):
        """Takes in the enum moveName and returns the set of frame inputs to perform that move"""
        return MovesDict[moveName]

    @staticmethod
    def getRandomMove():
        """Returns the name and frame inputs of a randomly selected move"""
        moveName = random.choice(list(Moves))
        moveInputs = MovesDict[moveName]
        return moveName, moveInputs

    @staticmethod
    def isDirectionalMove(move):
        """Determines if the selected move's inputs depend on the player's direction"""
        return move in [Moves.Fireball, Moves.HurricaneKick, Moves.DragonUppercut]

# Button indices: ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
MovesDict = {
    Moves.Idle: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves.Right: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    Moves.DownRight: [[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]],
    Moves.Down: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    Moves.DownLeft: [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]],
    Moves.Left: [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    Moves.UpLeft: [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]],
    Moves.Up: [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    Moves.UpRight: [[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]],
    Moves.LightPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],  # X
    Moves.MediumPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],  # Y
    Moves.HeavyPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],  # Z
    Moves.LightKick: [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],  # C
    Moves.MediumKick: [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # A
    Moves.HeavyKick: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # B
    Moves.CrouchLightPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],  # DOWN + X
    Moves.CrouchMediumPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]],  # DOWN + Y
    Moves.CrouchHeavyPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]],  # DOWN + Z
    Moves.CrouchLightKick: [[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]],  # DOWN + C
    Moves.CrouchMediumKick: [[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],  # DOWN + A
    Moves.CrouchHeavyKick: [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],  # DOWN + B
    Moves.LeftShoulderThrow: [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]],  # LEFT + X
    Moves.RightShoulderThrow: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]],  # RIGHT + X
    Moves.LeftSomersaultThrow: [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],  # LEFT + Z
    Moves.RightSomersaultThrow: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]],  # RIGHT + Z
    Moves.Fireball: [
        # Facing right: DOWN, DOWN+RIGHT, Y
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        # Facing left: DOWN, DOWN+LEFT, Y
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
    ],
    Moves.HurricaneKick: [
        # Facing right: DOWN, DOWN+LEFT, B
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        # Facing left: DOWN, DOWN+RIGHT, B
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ],
    Moves.DragonUppercut: [
        # Facing right: RIGHT, DOWN, Y
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        # Facing left: LEFT, DOWN, Y
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
    ],
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

class DeepQAgent:
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm to learn Street Fighter 2"""
    
    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    MAX_DATA_LENGTH = 10000
    DEFAULT_DISCOUNT_RATE = 0.98
    WIN_RATE_WINDOW = 10
    WIN_RATE_THRESHOLD = 0.1
    TIMESTEP_SCALE = 1000000  # Larger scale to reduce timestep impact

    # agent
    # 512 standing
    # 514 forward
    # 516 backward
    # 518 crounch
    # 520 jump up
    # 522 jump forward
    # 524 jump backward
    # 526 attack animation
    # 532 special move animation
    stateIndices = {512: 0, 514: 1, 516: 2, 518: 3, 520: 4, 522: 5, 524: 6, 526: 7, 532: 8}
    
    # enemy
    # 0 inactive
    # 528 knock down
    # 530 recover
    # 1024 stun
    # 1026 heavy be hit reaction
    # 1028 special move be hit reaction
    # 1030 throw
    # 1032 victory state
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]

    def __init__(self, stateSize=60, resume=False, epsilon=1.0, rl=0.001, name=None, moveList=Moves, lobby=None):
        """Initializes the agent and the underlying neural network"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.lobby = lobby
        self.learningRate = rl
        self.resume = resume
        self.memory = []
        self.model = self.initializeNetwork()

        # New memory for high-reward experiences
        self.high_reward_memory = []
        self.high_reward_threshold = 0.5  # Threshold to consider an experience as high-reward

        self.batch_size = 128
        self.target_update_freq = 100
        self.training_counter = 0
        
        self.epsilon = epsilon
        self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}
        
        if resume:
            self.loadModel()
            self.loadStats()
        
        self.total_timesteps = self.stats["total_timesteps"]
        
        if self.lobby and hasattr(self.lobby, 'training_stats'):
            if not self.lobby.training_stats:
                self.lobby.training_stats = {}
            if "wins" not in self.lobby.training_stats:
                self.lobby.training_stats["wins"] = self.stats.get("wins", 0)
            if "losses" not in self.lobby.training_stats:
                self.lobby.training_stats["losses"] = self.stats.get("losses", 0)
        
        self.saveStats()
        
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())

    def initializeNetwork(self):
        """Initializes a Neural Net with Dueling DQN architecture for improved performance"""
        input_layer = Input(shape=(self.stateSize,))
        shared = Dense(512, activation='relu')(input_layer)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.2)(shared)
        shared = Dense(384, activation='relu')(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.2)(shared)
        
        value_stream = Dense(256, activation='relu')(shared)
        value_stream = BatchNormalization()(value_stream)
        value_stream = Dense(128, activation='relu')(value_stream)
        value_stream = Dense(1)(value_stream)
        
        advantage_stream = Dense(256, activation='relu')(shared)
        advantage_stream = BatchNormalization()(advantage_stream)
        advantage_stream = Dense(128, activation='relu')(advantage_stream)
        advantage_stream = Dense(self.actionSize)(advantage_stream)
        
        advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage_stream)
        advantage_centered = Subtract()([advantage_stream, advantage_mean])
        q_values = Add()([value_stream, advantage_centered])
        
        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss=self._huber_loss, optimizer=Adam(learning_rate=self.learningRate))
        logger.info('Successfully initialized Dueling DQN model')
        return model

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss to use as the loss function for the model"""
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def loadStats(self):
        """Load stats from file if available"""
        stats_path = f"stats/{self.name}_stats.json"
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                    if "total_timesteps" not in self.stats:
                        self.stats["total_timesteps"] = 0
                logger.info(f"Loaded stats from {stats_path}: {self.stats}")
            except Exception as e:
                logger.error(f"Error loading stats from {stats_path}: {e}")
                self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}
        else:
            logger.info(f"No stats file found at {stats_path}. Using default stats.")
            self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}
        self.total_timesteps = self.stats["total_timesteps"]

    def saveStats(self):
        """Save stats to file"""
        os.makedirs("stats", exist_ok=True)
        stats_path = f"stats/{self.name}_stats.json"
        
        if self.lobby and hasattr(self.lobby, 'training_stats'):
            wins = self.lobby.training_stats.get("wins", 0)
            losses = self.lobby.training_stats.get("losses", 0)
        else:
            wins = self.stats.get("wins", 0)
            losses = self.stats.get("losses", 0)
        
        self.stats.update({
            "total_timesteps": self.total_timesteps,
            "wins": wins,
            "losses": losses
        })
        
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats to {stats_path}: {e}")

    def prepareNetworkInputs(self, step):
        feature_vector = []
        max_health = 100.0
        screen_width = 263.0
        screen_height = 240.0
        max_matches = 1.0
        max_score = 100000.0
        
        feature_vector.append(step.get("enemy_health", 100) / max_health)
        feature_vector.append(step.get("enemy_x_position", 0) / screen_width)
        feature_vector.append(step.get("enemy_y_position", 0) / screen_height)
        feature_vector.append(step.get("enemy_matches_won", 0) / max_matches)
        
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        enemy_status = step.get("enemy_status", 512)
        if enemy_status not in DeepQAgent.doneKeys:
            oneHotEnemyState[DeepQAgent.stateIndices[enemy_status]] = 1
        feature_vector += oneHotEnemyState
        
        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector += oneHotEnemyChar
        
        feature_vector.append(step.get("health", 100) / max_health)
        feature_vector.append(step.get("x_position", 0) / screen_width)
        feature_vector.append(step.get("y_position", 0) / screen_height)
        feature_vector.append(step.get("matches_won", 0) / max_matches)
        feature_vector.append(step.get("score", 0) / max_score)
        
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        player_status = step.get("status", 512)
        if player_status not in DeepQAgent.doneKeys:
            oneHotPlayerState[DeepQAgent.stateIndices[player_status]] = 1
        feature_vector += oneHotPlayerState
        
        player_x = step.get("x_position", 0)
        enemy_x = step.get("enemy_x_position", 0)
        x_distance = enemy_x - player_x
        feature_vector.append(x_distance / screen_width)
        
        feature_vector = feature_vector[:self.stateSize]
        if len(feature_vector) < self.stateSize:
            feature_vector += [0] * (self.stateSize - len(feature_vector))
        
        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def prepareForNextFight(self):
        """Reset memory for a new fight"""
        # Instead of completely clearing memory, keep some high-reward experiences
        if len(self.high_reward_memory) > 0:
            logger.info(f"Keeping {len(self.high_reward_memory)} high-reward experiences for continued learning")
            # Start with high-reward memories and add more as needed during play
            self.memory = self.high_reward_memory.copy()
        else:
            self.memory = []

    def getRandomMove(self, info):
        """Get a random move from the available move list"""
        move, frameInputs = Moves.getRandomMove()
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get("enemy_x_position", 200)
            frameInputs = frameInputs[0 if facing_right else 1]
        return move.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert a move to a list of button arrays."""
        frameInputs = Moves.getMoveInputs(move)
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get("enemy_x_position", 200)
            frameInputs = frameInputs[0 if facing_right else 1]
        return frameInputs

    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network"""
        start_time = time.time()
        
        if random.random() < self.epsilon:
            move_index, frameInputs = self.getRandomMove(info)
            logger.debug(f"Random move selected (exploration), epsilon: {self.epsilon:.4f}")
            return move_index, frameInputs
        
        stateData = self.prepareNetworkInputs(info)
        predictedRewards = self.model.predict(stateData, verbose=0)[0]
        move_index = np.argmax(predictedRewards)
        move = list(self.moveList)[move_index]
        frameInputs = self.convertMoveToFrameInputs(move, info)
        move_time = (time.time() - start_time) * 1000
        logger.debug(f"Predicted move selected in {move_time:.2f}ms, epsilon: {self.epsilon:.4f}")
        return move_index, frameInputs
    

    def recordStep(self, step):
        """Records a step with a simple reward structure and identifies high-reward experiences"""
        # Create a mutable copy of the step if it's a tuple
        if isinstance(step, tuple):
            step = list(step)
        
        # normalize rewards
        if step[self.REWARD_INDEX] != 0:
            step[self.REWARD_INDEX] = np.clip(step[self.REWARD_INDEX], -10, 10)
        
        # Add to memory
        self.memory.append(step)
        
        # Check if this is a high-reward experience
        if step[self.REWARD_INDEX] > self.high_reward_threshold:
            # Add to high-reward memory
            self.high_reward_memory.append(step)
            logger.debug(f"Added high-reward experience with reward {step[self.REWARD_INDEX]:.2f}")
            
            # Ensure high_reward_memory doesn't get too large
            if len(self.high_reward_memory) > DeepQAgent.MAX_DATA_LENGTH // 2:
                # Sort by reward and keep the highest
                self.high_reward_memory.sort(key=lambda x: x[self.REWARD_INDEX], reverse=True)
                self.high_reward_memory = self.high_reward_memory[:DeepQAgent.MAX_DATA_LENGTH // 4]
                logger.debug(f"Pruned high-reward memory to {len(self.high_reward_memory)} items")
            
        # Ensure memory doesn't get too large
        if len(self.memory) > DeepQAgent.MAX_DATA_LENGTH:
            self.memory.pop(0)
            
        self.total_timesteps += 1
        self.saveStats()

    def trainNetwork(self, data, model):
        """Runs through a training epoch reviewing the training data with sorted replay"""
        if not data:
            return model
        
        # Sort data by reward (highest first) to prioritize important experiences
        sorted_data = sorted(data, key=lambda x: x[self.REWARD_INDEX], reverse=True)
        
        # Take a mix of high-reward and random experiences
        high_reward_count = min(len(sorted_data), self.batch_size // 2)
        high_reward_samples = sorted_data[:high_reward_count]
        
        # Fill the rest with random samples to maintain diversity
        random_count = self.batch_size - high_reward_count
        if len(sorted_data) > high_reward_count:
            random_samples = random.sample(sorted_data[high_reward_count:], min(len(sorted_data) - high_reward_count, random_count))
        else:
            random_samples = []
        
        # Combine samples
        minibatch = high_reward_samples + random_samples
        
        # state size and action size
        states = np.zeros((len(minibatch), self.stateSize))  # Fixed: use tuple for shape
        targets = np.zeros((len(minibatch), self.actionSize))  # Fixed: use tuple for shape
        
        for i, (state, action, reward, done, next_state) in enumerate(minibatch):
            if isinstance(action, dict):
                action = action.get('value', 0)
            
            # pick action
            action = min(max(0, action), self.actionSize - 1)

            # store state
            states[i] = state

            target = model.predict(state, verbose=0)[0]
            
            if not done:
                # use target network for more stable learning
                next_q_values = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_q_values)
            else:
                target[action] = reward

            # store target
            targets[i] = target
            
        # out loop
        history = model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        self.training_counter += 1
        if self.training_counter % self.target_update_freq == 0:
            self.target_model.set_weights(model.get_weights())
            logger.info("target network updated")
        
        return model

    def reviewFight(self):
        """Review and learn from the previous fight, then save the model"""
        if self.memory:
            data = []
            for step in self.memory:
                state = self.prepareNetworkInputs(step[self.STATE_INDEX])
                action = step[self.ACTION_INDEX]
                reward = step[self.REWARD_INDEX]
                done = step[self.DONE_INDEX]
                next_state = self.prepareNetworkInputs(step[self.NEXT_STATE_INDEX])
                data.append([state, action, reward, done, next_state])

            # Train twice for better learning
            for _ in range(2):
                self.model = self.trainNetwork(data, self.model)
        
        # Save model periodically
        if self.total_timesteps % 1000 == 0:
            self.saveModel()
            logger.info("Model saved after fight")

        self.saveStats()

    def saveModel(self):
        """Save model weights to file with correct filename format"""
        try:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{self.name}Model.weights.h5"
            self.model.save_weights(model_path)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                logger.info(f"Model saved successfully to {model_path} (size: {file_size} bytes)")
            else:
                logger.error(f"Failed to save model: File {model_path} does not exist after save attempt")
            self.saveStats()
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def loadModel(self):
        """Load model weights from file if available"""
        model_path = f"models/{self.name}Model.h5"
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            logger.info(f"Loaded model from {model_path}")
            self.target_model.set_weights(self.model.get_weights())

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})