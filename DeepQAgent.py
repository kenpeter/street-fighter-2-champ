import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    Dropout,
    Lambda,
    Add,
    Subtract,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import logging
import json
from enum import Enum
import time
import math


# Define Moves Enum and MovesDict - REDUCED ACTION SPACE
class Moves(Enum):
    """Reduced action space - only essential moves like the 95% win rate version"""

    Idle = 0
    Right = 1
    Left = 2
    Down = 3
    LightPunch = 4
    MediumPunch = 5
    HeavyPunch = 6
    LightKick = 7
    MediumKick = 8
    HeavyKick = 9
    CrouchLightPunch = 10
    CrouchMediumPunch = 11
    # Removed complex moves that are hard to learn

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


# SIMPLIFIED MOVE DICTIONARY - Only essential moves
MovesDict = {
    Moves.Idle: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves.Right: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    Moves.Left: [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    Moves.Down: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    Moves.LightPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],  # X
    Moves.MediumPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],  # Y
    Moves.HeavyPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],  # Z
    Moves.LightKick: [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],  # C
    Moves.MediumKick: [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # A
    Moves.HeavyKick: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # B
    Moves.CrouchLightPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],  # DOWN + X
    Moves.CrouchMediumPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]],  # DOWN + Y
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")


class DeepQAgent:
    """An agent that implements the Deep Q Neural Network with EXACT 95% win rate reward system"""

    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    MAX_DATA_LENGTH = 10000
    DEFAULT_DISCOUNT_RATE = 0.94  # Same as PPO version

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

    def __init__(
        self,
        stateSize=60,
        total_timesteps=50000,
        name=None,
        moveList=Moves,
        lobby=None,
    ):
        """Initializes the agent with the corrected reward system"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)  # Now only 12 actions instead of 28
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.lobby = lobby

        # Timestep-based scheduling
        self.total_timesteps_target = total_timesteps
        self.current_timesteps = 0

        # CORRECTED: Use smaller learning rates like the PPO version
        self.initial_epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon = self.initial_epsilon

        self.initial_learning_rate = 2.5e-4  # Same as PPO start
        self.lr_min = 2.5e-6  # Same as PPO end
        self.learningRate = self.initial_learning_rate

        self.memory = []
        self.model = self.initializeNetwork()
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())

        self.batch_size = 512  # Same as PPO
        self.target_update_freq = 100
        self.training_counter = 0

        # Reward calculation setup - EXACT COPY of 95% version
        self.full_hp = 176
        self.reward_coeff = 3.0

        # Force model saving setup
        self.save_model_interval = 100000
        self.last_model_save = 0
        self.models_saved = 0

        logger.info(
            f"Agent initialized for {total_timesteps} timesteps with {self.actionSize} actions"
        )

    def update_parameters(self):
        """Update epsilon and learning rate - linear schedule like PPO"""
        progress = self.current_timesteps / self.total_timesteps_target

        # Linear decay for epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.initial_epsilon - (self.initial_epsilon - self.epsilon_min) * progress,
        )

        # Linear decay for learning rate (same as PPO)
        self.learningRate = max(
            self.lr_min,
            self.initial_learning_rate
            - (self.initial_learning_rate - self.lr_min) * progress,
        )

    def initializeNetwork(self):
        """Simplified network architecture"""
        input_layer = Input(shape=(self.stateSize,))

        # Smaller network for reduced action space
        shared = Dense(64, activation="relu")(input_layer)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.1)(shared)
        shared = Dense(32, activation="relu")(shared)
        shared = BatchNormalization()(shared)

        value_stream = Dense(16, activation="relu")(shared)
        value_stream = Dense(1)(value_stream)

        advantage_stream = Dense(16, activation="relu")(shared)
        advantage_stream = Dense(self.actionSize)(advantage_stream)

        advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(
            advantage_stream
        )
        advantage_centered = Subtract()([advantage_stream, advantage_mean])
        q_values = Add()([value_stream, advantage_centered])

        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(
            loss=self._huber_loss, optimizer=Adam(learning_rate=self.learningRate)
        )
        return model

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss"""
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (
            K.abs(error) - clip_delta
        )
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def prepareNetworkInputs(self, step):
        """Prepare network inputs from game state"""
        feature_vector = []
        max_health = 176.0
        screen_width = 320.0
        screen_height = 240.0
        max_matches = 1.0
        max_score = 100000.0

        # Basic normalized features
        feature_vector.append(step.get("enemy_health", 176) / max_health)
        feature_vector.append(step.get("enemy_x_position", 200) / screen_width)
        feature_vector.append(step.get("enemy_y_position", 0) / screen_height)
        feature_vector.append(step.get("enemy_matches_won", 0) / max_matches)

        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        enemy_status = step.get("enemy_status", 512)
        if (
            enemy_status not in DeepQAgent.doneKeys
            and enemy_status in DeepQAgent.stateIndices
        ):
            oneHotEnemyState[DeepQAgent.stateIndices[enemy_status]] = 1
        feature_vector += oneHotEnemyState

        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if 0 <= enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector += oneHotEnemyChar

        feature_vector.append(step.get("health", 176) / max_health)
        feature_vector.append(step.get("x_position", 100) / screen_width)
        feature_vector.append(step.get("y_position", 0) / screen_height)
        feature_vector.append(step.get("matches_won", 0) / max_matches)
        feature_vector.append(step.get("score", 0) / max_score)

        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        player_status = step.get("status", 512)
        if (
            player_status not in DeepQAgent.doneKeys
            and player_status in DeepQAgent.stateIndices
        ):
            oneHotPlayerState[DeepQAgent.stateIndices[player_status]] = 1
        feature_vector += oneHotPlayerState

        # Distance feature
        player_x = step.get("x_position", 100)
        enemy_x = step.get("enemy_x_position", 200)
        x_distance = (enemy_x - player_x) / screen_width
        feature_vector.append(x_distance)

        # Pad to exact size
        feature_vector = feature_vector[: self.stateSize]
        if len(feature_vector) < self.stateSize:
            feature_vector += [0] * (self.stateSize - len(feature_vector))

        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def prepareForNextFight(self):
        """Reset memory for next fight"""
        self.memory = []

    def getRandomMove(self, info):
        """Get a random move from reduced action space"""
        move, frameInputs = Moves.getRandomMove()
        return move.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert move to frame inputs"""
        frameInputs = Moves.getMoveInputs(move)
        return frameInputs

    def getMove(self, obs, info):
        """Returns button inputs with epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            move_index, frameInputs = self.getRandomMove(info)
            return move_index, frameInputs

        stateData = self.prepareNetworkInputs(info)
        predictedRewards = self.model.predict(stateData, verbose=0)[0]
        move_index = np.argmax(predictedRewards)
        move = list(self.moveList)[move_index]
        frameInputs = self.convertMoveToFrameInputs(move, info)

        return move_index, frameInputs

    def calculate_exact_reward(
        self,
        prev_player_health,
        prev_opponent_health,
        curr_player_health,
        curr_opponent_health,
    ):
        """
        EXACT COPY of the 95% win rate reward calculation
        """
        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(
                self.full_hp, (curr_opponent_health + 1) / (self.full_hp + 1)
            )
            custom_done = True

        # Game is over and player wins.
        elif curr_opponent_health < 0:
            custom_reward = (
                math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1))
                * self.reward_coeff
            )
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (
                prev_opponent_health - curr_opponent_health
            ) - (prev_player_health - curr_player_health)
            custom_done = False

        # CRITICAL: Apply the exact same reward normalization as the 95% version
        normalized_reward = 0.001 * custom_reward

        return normalized_reward, custom_done

    def recordStep(self, step):
        """Record a step with the EXACT reward calculation from 95% win rate version"""
        if isinstance(step, tuple):
            step = list(step)

        # Extract health information
        prev_info = step[self.STATE_INDEX]
        curr_info = step[self.NEXT_STATE_INDEX]

        prev_player_health = prev_info.get("health", 176)
        prev_opponent_health = prev_info.get("enemy_health", 176)
        curr_player_health = curr_info.get("health", 176)
        curr_opponent_health = curr_info.get("enemy_health", 176)

        # Use the EXACT same reward calculation as the 95% win rate version
        normalized_reward, custom_done = self.calculate_exact_reward(
            prev_player_health,
            prev_opponent_health,
            curr_player_health,
            curr_opponent_health,
        )

        # Replace the reward with the corrected calculation
        step[self.REWARD_INDEX] = normalized_reward

        # Update done status if needed
        if custom_done:
            step[self.DONE_INDEX] = True

        # Add to memory (no additional clipping - already normalized)
        self.memory.append(step)

        # Keep memory size manageable
        if len(self.memory) > DeepQAgent.MAX_DATA_LENGTH:
            self.memory.pop(0)

        self.current_timesteps += 1

        # Update parameters based on progress
        self.update_parameters()

        # FORCE MODEL SAVING DURING TRAINING
        if self.current_timesteps - self.last_model_save >= self.save_model_interval:
            logger.info(f"💾 Auto-saving model at timestep {self.current_timesteps}")
            success = self.saveModel()
            if success:
                self.last_model_save = self.current_timesteps
                self.models_saved += 1
                logger.info(f"✅ Model #{self.models_saved} saved successfully!")
            else:
                logger.error(
                    f"❌ Model save failed at timestep {self.current_timesteps}"
                )

    def trainNetwork(self, data, model):
        """Train the network on collected data"""
        if not data or len(data) < 32:
            return model

        # Random sampling
        batch_size = min(self.batch_size, len(data))
        minibatch = random.sample(data, batch_size)

        states = np.zeros((len(minibatch), self.stateSize))
        targets = np.zeros((len(minibatch), self.actionSize))

        for i, (state, action, reward, done, next_state) in enumerate(minibatch):
            if isinstance(action, dict):
                action = action.get("value", 0)
            action = min(max(0, action), self.actionSize - 1)

            states[i] = state
            target = model.predict(state, verbose=0)[0]

            if not done:
                next_q_values = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_q_values)
            else:
                target[action] = reward
            targets[i] = target

        # Update learning rate
        try:
            if hasattr(model.optimizer, "learning_rate"):
                if hasattr(model.optimizer.learning_rate, "assign"):
                    model.optimizer.learning_rate.assign(float(self.learningRate))
                else:
                    K.set_value(model.optimizer.learning_rate, float(self.learningRate))
        except (AttributeError, TypeError, ValueError):
            pass

        # Train the model
        model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        self.training_counter += 1
        if self.training_counter % self.target_update_freq == 0:
            self.target_model.set_weights(model.get_weights())

        return model

    def reviewFight(self):
        """Review fight and train on collected experience"""
        if self.memory:
            try:
                data = []
                for step in self.memory:
                    state = self.prepareNetworkInputs(step[self.STATE_INDEX])
                    action = step[self.ACTION_INDEX]
                    reward = step[self.REWARD_INDEX]
                    done = step[self.DONE_INDEX]
                    next_state = self.prepareNetworkInputs(step[self.NEXT_STATE_INDEX])
                    data.append([state, action, reward, done, next_state])

                self.model = self.trainNetwork(data, self.model)

                # Save model at the end
                logger.info("💾 Script ending - saving final model...")
                success = self.saveModel()
                if success:
                    logger.info(f"✅ Final model saved successfully!")
                else:
                    logger.error(f"❌ Final model save failed!")

            except Exception as e:
                logger.error(f"❌ Error in reviewFight: {e}")
                try:
                    self.saveModel()
                except Exception as save_error:
                    logger.error(f"❌ Emergency save failed: {save_error}")

    def saveModel(self):
        """Save model using only weights format"""
        try:
            os.makedirs("models", exist_ok=True)
            weights_path = f"models/DeepQAgentModel_{self.current_timesteps}.weights.h5"

            self.model.save_weights(weights_path)

            if os.path.exists(weights_path):
                file_size = os.path.getsize(weights_path)
                logger.info(
                    f"✅ Model weights saved: {weights_path} ({file_size:,} bytes)"
                )
                return True
            else:
                logger.error(f"❌ Weights file not created: {weights_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Error in saveModel: {e}")
            return False


# Register custom loss function
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})
