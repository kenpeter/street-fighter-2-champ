import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import logging
from enum import Enum

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

# Placeholder for LossHistory
class LossHistory:
    def __init__(self):
        self.losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
    
    def losses_clear(self):
        self.losses = []

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
    EPSILON_MIN = 0.1
    DEFAULT_EPSILON_DECAY = 0.999
    DEFAULT_DISCOUNT_RATE = 0.98
    DEFAULT_LEARNING_RATE = 0.0001

    stateIndices = {512: 0, 514: 1, 516: 2, 518: 3, 520: 4, 522: 5, 524: 6, 526: 7, 532: 8}
    doneKeys = [0, 528, 530, 1024, 1026, 1028, 1030, 1032]

    def __init__(self, stateSize=40, resume=False, epsilon=1, name=None, moveList=Moves):
        """Initializes the agent and the underlying neural network"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        
        self.epsilon = DeepQAgent.EPSILON_MIN if resume else epsilon
        self.epsilonDecay = DeepQAgent.DEFAULT_EPSILON_DECAY
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.memory = []
        self.model = self.initializeNetwork()
        
        if resume:
            self.loadModel()
            
        self.target_model = self.initializeNetwork()
        self.target_model.set_weights(self.model.get_weights())
        self.total_timesteps = 0

    def initializeNetwork(self):
        """Initializes a Neural Net for a Deep-Q learning Model"""
        model = Sequential([
            Input(shape=(self.stateSize,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.actionSize, activation='linear')
        ])
        
        model.compile(loss=self._huber_loss, optimizer=Adam(learning_rate=self.learningRate))
        logger.info('Successfully initialized model')
        return model

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss to use as the loss function for the model"""
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def prepareNetworkInputs(self, step):
        """Generates a feature vector from the current game state information to feed into the network"""
        feature_vector = []
        
        # Enemy Data
        feature_vector.append(step.get("enemy_health", 100))
        feature_vector.append(step.get("enemy_x_position", 0))
        feature_vector.append(step.get("enemy_y_position", 0))

        # One-hot encode enemy state
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        enemy_status = step.get("enemy_status", 512)
        if enemy_status not in DeepQAgent.doneKeys:
            oneHotEnemyState[DeepQAgent.stateIndices[enemy_status]] = 1
        feature_vector += oneHotEnemyState

        # One-hot encode enemy character
        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector += oneHotEnemyChar

        # Player Data
        feature_vector.append(step.get("health", 100))
        feature_vector.append(step.get("x_position", 0))
        feature_vector.append(step.get("y_position", 0))

        # One-hot encode player state
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        player_status = step.get("status", 512)
        if player_status not in DeepQAgent.doneKeys:
            oneHotPlayerState[DeepQAgent.stateIndices[player_status]] = 1
        feature_vector += oneHotPlayerState

        # Ensure feature_vector length matches stateSize (40)
        feature_vector = feature_vector[:self.stateSize]
        if len(feature_vector) < self.stateSize:
            feature_vector += [0] * (self.stateSize - len(feature_vector))
        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def prepareForNextFight(self):
        """Reset memory for a new fight"""
        self.memory = []

    def getRandomMove(self, info):
        """Get a random move from the available move list"""
        move, frameInputs = Moves.getRandomMove()
        # Handle directional moves
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get("enemy_x_position", 200)
            frameInputs = frameInputs[0 if facing_right else 1]
        return move.value, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert a move to a list of button arrays."""
        frameInputs = Moves.getMoveInputs(move)
        # Handle directional moves
        if Moves.isDirectionalMove(move):
            facing_right = info.get("x_position", 100) < info.get("enemy_x_position", 200)
            frameInputs = frameInputs[0 if facing_right else 1]
        return frameInputs

    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network"""
        if np.random.rand() <= self.epsilon:
            # Explore: choose random action
            move_index, frameInputs = self.getRandomMove(info)
            return move_index, frameInputs
        else:
            # Exploit: choose best action according to model
            stateData = self.prepareNetworkInputs(info)
            predictedRewards = self.model.predict(stateData, verbose=0)[0]
            move_index = np.argmax(predictedRewards)
            move = list(self.moveList)[move_index]
            frameInputs = self.convertMoveToFrameInputs(move, info)
            return move_index, frameInputs

    def recordStep(self, step):
        """Records a step with a simple reward structure"""
        self.memory.append(step)
        
        if len(self.memory) > DeepQAgent.MAX_DATA_LENGTH:
            self.memory.pop(0)
            
        self.total_timesteps += 1

    def trainNetwork(self, data, model):
        """Runs through a training epoch reviewing the training data"""
        if not data:
            return model
        
        minibatch = random.sample(data, min(len(data), 32))
        self.lossHistory.losses_clear()
        
        for state, action, reward, done, next_state in minibatch:
            if isinstance(action, dict):
                action = action.get('value', 0)
            
            action = min(max(0, action), self.actionSize - 1)
            
            modelOutput = model.predict(state, verbose=0)[0]
            
            if not done:
                reward = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            modelOutput[action] = reward
            modelOutput = np.reshape(modelOutput, [1, self.actionSize])
            model.fit(state, modelOutput, epochs=1, verbose=0, callbacks=[self.lossHistory])
        
        return model

    def reviewFight(self):
        """Review and learn from the previous fight"""
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
        
        if self.total_timesteps % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.epsilonDecay

    def saveModel(self):
        """Save model weights to file"""
        os.makedirs("models", exist_ok=True)
        self.model.save_weights(f"models/{self.name}Model.h5")

    def loadModel(self):
        """Load model weights from file if available"""
        model_path = f"models/{self.name}Model.h5"
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            logger.info(f"Loaded model from {model_path}")
            self.target_model.set_weights(self.model.get_weights())

# Register custom loss function
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})

if __name__ == "__main__":
    agent = DeepQAgent(stateSize=40)
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
                "enemy_character": 0,
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
                "enemy_character": state["enemy_character"],
            }
            reward = 1.0 if next_state["enemy_health"] < state["enemy_health"] else -0.5
            done = next_state["health"] <= 0 or next_state["enemy_health"] <= 0
            agent.recordStep([None, state, move, reward, None, next_state, done])
            if done:
                break
        agent.reviewFight()
        logger.info(f"Episode {episode+1} completed. Epsilon: {agent.epsilon:.4f}")
    agent.saveModel()