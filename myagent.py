import numpy as np
import random
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras.losses
from LossHistory import LossHistory
from DefaultMoveList import Moves
import os
import logging
from keras.layers import Input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

class DeepQAgent:
    """An agent that implements the Deep Q Neural Network Reinforcement Algorithm to learn street fighter 2"""
    
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
    ACTION_BUTTONS = ['X', 'Y', 'Z', 'A', 'B', 'C']

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        """Implementation of huber loss to use as the loss function for the model"""
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def __init__(self, stateSize=32, resume=False, epsilon=1, name=None, moveList=Moves):
        """Initializes the agent and the underlying neural network"""
        self.name = name or self.__class__.__name__
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE
        
        # Use resume parameter instead of load
        self.epsilon = DeepQAgent.EPSILON_MIN if resume else epsilon
        self.epsilonDecay = DeepQAgent.DEFAULT_EPSILON_DECAY
        self.learningRate = DeepQAgent.DEFAULT_LEARNING_RATE
        self.lossHistory = LossHistory()
        self.memory = []
        self.model = self.initializeNetwork()
        
        # Load model if resuming
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
        
        # Updated parameter name
        model.compile(loss=DeepQAgent._huber_loss, optimizer=Adam(learning_rate=self.learningRate))
        
        print('Successfully initialized model')
        return model

    def prepareNetworkInputs(self, step):
        """Generates a feature vector from the current game state information to feed into the network"""
        feature_vector = []
        
        # Enemy Data
        feature_vector.append(step.get("enemy_health", 100))
        feature_vector.append(step.get("enemy_x_position", 0))
        feature_vector.append(step.get("enemy_y_position", 0))

        # one hot encode enemy state
        oneHotEnemyState = [0] * len(DeepQAgent.stateIndices.keys())
        enemy_status = step.get("enemy_status", 512)
        if enemy_status not in DeepQAgent.doneKeys:
            oneHotEnemyState[DeepQAgent.stateIndices[enemy_status]] = 1
        feature_vector += oneHotEnemyState

        # one hot encode enemy character
        oneHotEnemyChar = [0] * 8
        enemy_char = step.get("enemy_character", 0)
        if enemy_char < len(oneHotEnemyChar):
            oneHotEnemyChar[enemy_char] = 1
        feature_vector += oneHotEnemyChar

        # Player Data
        feature_vector.append(step.get("health", 100))
        feature_vector.append(step.get("x_position", 0))
        feature_vector.append(step.get("y_position", 0))

        # one hot encode player state
        oneHotPlayerState = [0] * len(DeepQAgent.stateIndices.keys())
        player_status = step.get("status", 512)
        if player_status not in DeepQAgent.doneKeys:
            oneHotPlayerState[DeepQAgent.stateIndices[player_status]] = 1
        feature_vector += oneHotPlayerState

        feature_vector = np.reshape(feature_vector, [1, self.stateSize])
        return feature_vector

    def prepareForNextFight(self):
        """Reset memory for a new fight"""
        self.memory = []

    def getRandomMove(self, info):
        """Get a random move from the available move list"""
        moveName = random.choice(list(self.moveList)) if self.moveList else 0
        frameInputs = self.convertMoveToFrameInputs(moveName, info)
        return moveName.value if hasattr(moveName, 'value') else moveName, frameInputs

    def convertMoveToFrameInputs(self, move, info):
        """Convert a move to frame inputs"""
        # Simple implementation that maps moves to basic inputs
        # Assuming moveList contains names that can be mapped to button presses
        frameInputs = []
        
        # Generate a simple sequence based on move name
        if hasattr(move, 'name'):
            move_name = move.name
        else:
            move_name = str(move)
        
        # Basic mapping for demonstration
        if 'PUNCH' in move_name:
            frameInputs = ['X']
        elif 'KICK' in move_name:
            frameInputs = ['B']
        elif 'SPECIAL' in move_name:
            frameInputs = ['→', '↓', '↘', 'Y']
        else:
            # Default to a simple input if no match
            frameInputs = [random.choice(self.ACTION_BUTTONS)]
        
        return frameInputs


    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network"""
        if np.random.rand() <= self.epsilon:
            move, frameInputs = self.getRandomMove(info)
            return move, frameInputs  # Return the full tuple
        else:
            stateData = self.prepareNetworkInputs(info)
            predictedRewards = self.model.predict(stateData, verbose=0)[0]
            move_index = np.argmax(predictedRewards)
            move = list(self.moveList)[move_index]
            frameInputs = self.convertMoveToFrameInputs(move, info)
            return move, frameInputs  # Return the full tuple
    

    def recordStep(self, step):
        """Records a step with simplified reward structure"""
        current_state = step[DeepQAgent.STATE_INDEX]
        next_state = step[DeepQAgent.NEXT_STATE_INDEX]
        reward = step[DeepQAgent.REWARD_INDEX]
        action = step[DeepQAgent.ACTION_INDEX]
        done = step[DeepQAgent.DONE_INDEX]

        current_player_health = current_state.get("health", 100) if current_state else 100
        current_enemy_health = current_state.get("enemy_health", 100) if current_state else 100
        next_player_health = next_state.get("health", 100) if next_state else 100
        next_enemy_health = next_state.get("enemy_health", 100) if next_state else 100

        damage_dealt = max(0, current_enemy_health - next_enemy_health)
        damage_taken = max(0, current_player_health - next_player_health)

        modified_reward = reward
        modified_reward += damage_dealt * 0.1
        modified_reward -= damage_taken * 0.1
        if next_enemy_health <= 0:
            modified_reward += 10.0
        elif next_player_health <= 0:
            modified_reward -= 10.0

        modified_step = list(step)
        modified_step[DeepQAgent.REWARD_INDEX] = modified_reward
        self.memory.append(modified_step)
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
            # Fix the action handling
            if isinstance(action, dict):
                # Simpler action extraction
                action = action.get('value', 0)
            
            # Ensure action is within valid range
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
                # Unpack using the correct indices
                state = self.prepareNetworkInputs(step[self.STATE_INDEX])
                action = step[self.ACTION_INDEX]
                reward = step[self.REWARD_INDEX]
                done = step[self.DONE_INDEX]
                next_state = self.prepareNetworkInputs(step[self.NEXT_STATE_INDEX])
                data.append([state, action, reward, done, next_state])
            self.model = self.trainNetwork(data, self.model)
        
        # Update target network less frequently for stability
        if self.total_timesteps % 100 == 0:  # Changed from 8 to 100
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
            print(f"Loaded model from {model_path}")
            self.target_model.set_weights(self.model.get_weights())

try:
    # For newer TensorFlow/Keras versions
    from keras.utils import get_custom_objects
except ImportError:
    try:
        # For older TensorFlow/Keras versions
        from keras.utils.generic_utils import get_custom_objects
    except ImportError:
        # As a last resort, try this path
        from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"_huber_loss": DeepQAgent._huber_loss})

if __name__ == "__main__":
    agent = DeepQAgent(stateSize=32, load=False)
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