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
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm"""

    OBSERVATION_INDEX = 0
    STATE_INDEX = 1
    ACTION_INDEX = 2
    REWARD_INDEX = 3
    NEXT_OBSERVATION_INDEX = 4
    NEXT_STATE_INDEX = 5
    DONE_INDEX = 6
    MAX_DATA_LENGTH = 10000
    DEFAULT_DISCOUNT_RATE = 0.98

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
        """Initializes the agent with timestep-based scheduling"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        self.lobby = lobby

        # Timestep-based scheduling
        self.total_timesteps_target = total_timesteps
        self.current_timesteps = 0

        # Calculate epsilon and learning rate based on single script run
        self.initial_epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon = self.initial_epsilon

        self.initial_learning_rate = 0.001
        self.lr_min = 0.0001
        self.learningRate = self.initial_learning_rate

        self.memory = []
        self.model = self.initializeNetwork()
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())

        self.batch_size = 512
        self.target_update_freq = 100
        self.training_counter = 0

        logger.info(f"Agent initialized for {total_timesteps} timesteps")

    def update_parameters(self):
        """Update epsilon and learning rate based on current progress"""
        progress = self.current_timesteps / self.total_timesteps_target

        # Linear decay for epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.initial_epsilon - (self.initial_epsilon - self.epsilon_min) * progress,
        )

        # Exponential decay for learning rate
        self.learningRate = max(
            self.lr_min,
            self.initial_learning_rate
            * (0.5 ** (progress * 5)),  # Halve 5 times over the run
        )

    def initializeNetwork(self):
        """Initializes a small Neural Net with Dueling DQN architecture"""
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
        """Reset memory for next fight"""
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

    def recordStep(self, step):
        """Record a step in memory"""
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

        self.current_timesteps += 1

        # Update parameters based on progress
        self.update_parameters()

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

        # Update optimizer learning rate
        K.set_value(model.optimizer.learning_rate, self.learningRate)

        # Train the model
        model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        self.training_counter += 1
        if self.training_counter % self.target_update_freq == 0:
            self.target_model.set_weights(model.get_weights())

        return model

    def reviewFight(self):
        """Review fight and train on collected experience"""
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


# Register custom loss function
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})
