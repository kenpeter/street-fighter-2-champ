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


# Define Moves Enum and MovesDict (keep your existing moves code)
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
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
        # Facing left: DOWN, DOWN+LEFT, Y
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
    ],
    Moves.HurricaneKick: [
        # Facing right: DOWN, DOWN+LEFT, B
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        # Facing left: DOWN, DOWN+RIGHT, B
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    Moves.DragonUppercut: [
        # Facing right: RIGHT, DOWN, Y
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
        # Facing left: LEFT, DOWN, Y
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
    ],
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")


class DeepQAgent:
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm with AUTO SCHEDULING"""

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
    TIMESTEP_SCALE = 1000000

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
        resume=False,
        name=None,
        moveList=Moves,
        lobby=None,
    ):
        """Initializes the agent with AUTOMATIC scheduling - no more manual epsilon/lr!"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.lobby = lobby

        # AUTO SCHEDULING PARAMETERS - SUPER SIMPLE!
        self.initial_epsilon = 1.0
        self.epsilon = 1.0  # Will be overridden if resuming
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay for better exploration

        self.initial_learning_rate = 0.001
        self.learningRate = 0.001  # Will be overridden if resuming
        self.lr_min = 0.0001
        self.lr_decay_every = 10000  # Every 10k steps
        self.lr_decay_factor = 0.9

        self.resume = resume
        self.memory = []

        # Load stats FIRST to get current epsilon/lr values
        self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}
        if resume:
            self.loadStats()  # This sets epsilon and learningRate from saved values

        self.model = self.initializeNetwork()  # Now uses correct learningRate

        # Simple memory - no complex high-reward stuff

        self.batch_size = 512
        self.target_update_freq = 100
        self.training_counter = 0

        if resume:
            self.loadModel()

        self.total_timesteps = self.stats["total_timesteps"]

        if self.lobby and hasattr(self.lobby, "training_stats"):
            if not self.lobby.training_stats:
                self.lobby.training_stats = {}
            if "wins" not in self.lobby.training_stats:
                self.lobby.training_stats["wins"] = self.stats.get("wins", 0)
            if "losses" not in self.lobby.training_stats:
                self.lobby.training_stats["losses"] = self.stats.get("losses", 0)

        self.saveStats()

        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())

        # Add these for less frequent saving
        self.save_stats_interval = 1000
        self.save_model_interval = 5000
        self.last_stats_save = 0
        self.last_model_save = 0

    def auto_schedule_params(self):
        """AUTOMATIC parameter scheduling - called every step"""

        # AUTO EPSILON DECAY - gets more aggressive over time
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # AUTO LEARNING RATE DECAY - every 10k steps
        if self.total_timesteps > 0 and self.total_timesteps % self.lr_decay_every == 0:
            old_lr = self.learningRate
            self.learningRate = max(
                self.lr_min, self.learningRate * self.lr_decay_factor
            )
            if old_lr != self.learningRate:
                # Update the model's learning rate
                K.set_value(self.model.optimizer.learning_rate, self.learningRate)
                logger.info(
                    f"🔄 AUTO: Learning rate {old_lr:.6f} → {self.learningRate:.6f} at step {self.total_timesteps}"
                )

        # Log epsilon changes every 5000 steps
        if self.total_timesteps % 5000 == 0 and self.total_timesteps > 0:
            logger.info(
                f"🎯 AUTO: Epsilon={self.epsilon:.4f}, LR={self.learningRate:.6f} at step {self.total_timesteps}"
            )

    def initializeNetwork(self):
        """Initializes a very small Neural Net with Dueling DQN architecture"""
        input_layer = Input(shape=(self.stateSize,))
        shared = Dense(64, activation="relu")(input_layer)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.1)(shared)
        shared = Dense(32, activation="relu")(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.1)(shared)

        value_stream = Dense(16, activation="relu")(shared)
        value_stream = BatchNormalization()(value_stream)
        value_stream = Dense(8, activation="relu")(value_stream)
        value_stream = Dense(1)(value_stream)

        advantage_stream = Dense(16, activation="relu")(shared)
        advantage_stream = BatchNormalization()(advantage_stream)
        advantage_stream = Dense(8, activation="relu")(advantage_stream)
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
        logger.info("✅ AUTO-SCHEDULED DQN initialized (64-32-16-8 neurons)")
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

    def loadStats(self):
        """Load stats from file if available"""
        stats_path = f"stats/{self.name}_stats.json"
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                    if "total_timesteps" not in self.stats:
                        self.stats["total_timesteps"] = 0

                    # 🔥 CRITICAL: Restore epsilon and learning rate from saved values
                    if "current_epsilon" in loaded_stats:
                        self.epsilon = loaded_stats["current_epsilon"]
                        logger.info(
                            f"🎯 RESUMED: Epsilon restored to {self.epsilon:.4f}"
                        )

                    if "current_learning_rate" in loaded_stats:
                        self.learningRate = loaded_stats["current_learning_rate"]
                        logger.info(
                            f"📚 RESUMED: Learning rate restored to {self.learningRate:.6f}"
                        )

                logger.info(
                    f"📊 Loaded stats: Steps={self.stats['total_timesteps']}, Wins={self.stats.get('wins',0)}, Losses={self.stats.get('losses',0)}"
                )

            except Exception as e:
                logger.error(f"Error loading stats: {e}")
                self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}
                logger.info("⚠️  Starting fresh due to load error")
        else:
            logger.info(
                "🆕 No previous stats found. Starting fresh with epsilon=1.0, lr=0.001"
            )
            self.stats = {"total_timesteps": 0, "wins": 0, "losses": 0}

        self.total_timesteps = self.stats["total_timesteps"]

    def saveStats(self):
        """Save stats including current scheduling parameters"""
        os.makedirs("stats", exist_ok=True)
        stats_path = f"stats/{self.name}_stats.json"

        if self.lobby and hasattr(self.lobby, "training_stats"):
            wins = self.lobby.training_stats.get("wins", 0)
            losses = self.lobby.training_stats.get("losses", 0)
        else:
            wins = self.stats.get("wins", 0)
            losses = self.stats.get("losses", 0)

        # Save current scheduling state
        self.stats.update(
            {
                "total_timesteps": self.total_timesteps,
                "wins": wins,
                "losses": losses,
                "current_epsilon": self.epsilon,
                "current_learning_rate": self.learningRate,
                "initial_epsilon": self.initial_epsilon,
                "initial_learning_rate": self.initial_learning_rate,
            }
        )

        try:
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

    def prepareNetworkInputs(self, step):
        """Same as before - prepare network inputs"""
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

        feature_vector = feature_vector[: self.stateSize]
        if len(feature_vector) < self.stateSize:
            feature_vector += [0] * (self.stateSize - len(feature_vector))

        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def prepareForNextFight(self):
        """Simple memory reset"""
        self.memory = []

    def getRandomMove(self, info):
        """Get a random move"""
        move, frameInputs = Moves.getRandomMove()
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get(
                "enemy_x_position", 200
            )
            frameInputs = frameInputs[0 if facing_right else 1]
        return move.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert move to frame inputs"""
        frameInputs = Moves.getMoveInputs(move)
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get(
                "enemy_x_position", 200
            )
            frameInputs = frameInputs[0 if facing_right else 1]
        return frameInputs

    def getMove(self, obs, info):
        """Returns button inputs with AUTO SCHEDULING"""
        start_time = time.time()

        if random.random() < self.epsilon:
            move_index, frameInputs = self.getRandomMove(info)
            return move_index, frameInputs

        stateData = self.prepareNetworkInputs(info)
        predictedRewards = self.model.predict(stateData, verbose=0)[0]
        move_index = np.argmax(predictedRewards)
        move = list(self.moveList)[move_index]
        frameInputs = self.convertMoveToFrameInputs(move, info)

        return move_index, frameInputs

    def recordStep(self, step):
        """Simple step recording with AUTO SCHEDULING"""
        if isinstance(step, tuple):
            step = list(step)

        # Clip rewards
        if step[self.REWARD_INDEX] != 0:
            step[self.REWARD_INDEX] = np.clip(step[self.REWARD_INDEX], -10, 10)

        # Add to memory
        self.memory.append(step)

        # Keep memory size manageable
        if len(self.memory) > DeepQAgent.MAX_DATA_LENGTH:
            self.memory.pop(0)

        self.total_timesteps += 1

        # 🚀 AUTO SCHEDULING HAPPENS HERE!
        self.auto_schedule_params()

        # Save stats occasionally
        if self.total_timesteps - self.last_stats_save >= self.save_stats_interval:
            self.saveStats()
            self.last_stats_save = self.total_timesteps

    def trainNetwork(self, data, model):
        """Simple training - just random sampling from memory"""
        if not data or len(data) < 32:
            return model

        # Simple random sampling
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

        # Train the model
        model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        self.training_counter += 1
        if self.training_counter % self.target_update_freq == 0:
            self.target_model.set_weights(model.get_weights())
            logger.info("🎯 Target network updated")

        return model

    def reviewFight(self):
        """Review fight and train"""
        if self.memory:
            data = []
            for step in self.memory:
                state = self.prepareNetworkInputs(step[self.STATE_INDEX])
                action = step[self.ACTION_INDEX]
                reward = step[self.REWARD_INDEX]
                done = step[self.DONE_INDEX]
                next_state = self.prepareNetworkInputs(step[self.NEXT_STATE_INDEX])
                data.append([state, action, reward, done, next_state])

            self.model = self.trainNetwork(data, self.model)

        if self.total_timesteps - self.last_model_save >= self.save_model_interval:
            self.saveModel()
            self.last_model_save = self.total_timesteps
            logger.info(f"💾 Model saved at timestep {self.total_timesteps}")

    def saveModel(self):
        """Save model"""
        try:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{self.name}Model.weights.h5"
            self.model.save_weights(model_path)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                logger.info(f"💾 Model saved to {model_path} ({file_size} bytes)")
            self.saveStats()
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def loadModel(self):
        """Load model"""
        model_path = f"models/{self.name}Model.weights.h5"
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"📂 Loaded model from {model_path}")
                self.target_model.set_weights(self.model.get_weights())
            except Exception as e:
                logger.error(f"Error loading model: {e}")


# Register custom loss function
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})
