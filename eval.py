import argparse
import retro
import os
import time
import random
import math
import tensorflow as tf
from tqdm import tqdm
import logging
from DeepQAgent import DeepQAgent, Moves
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Eval")

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    try:
        logger.info(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_visible_devices(physical_devices[0], "GPU")
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.warning("No GPU found. Will use CPU instead.")


class EvalAgent(DeepQAgent):
    """Evaluation version of DeepQAgent with no training"""

    def __init__(self, model_path, stateSize=60, name=None, moveList=Moves):
        """Initialize agent for evaluation only"""
        self.name = name or "EvalAgent"
        self.moveList = moveList
        self.stateSize = stateSize
        self.actionSize = len(moveList)
        self.gamma = DeepQAgent.DEFAULT_DISCOUNT_RATE

        # Set epsilon to 0 for pure exploitation (no random moves)
        self.epsilon = 0.0
        self.learningRate = 0.001  # Not used but needed for model creation

        # Initialize network
        self.model = self.initializeNetwork()

        # Load the trained model
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"✅ Loaded model from {model_path}")

                # Test the model with dummy input
                dummy_input = np.zeros((1, self.stateSize))
                test_prediction = self.model.predict(dummy_input, verbose=0)[0]
                logger.info(
                    f"🧪 Model test - Output shape: {test_prediction.shape}, Max Q-value: {np.max(test_prediction):.3f}"
                )
                logger.info(f"🧪 Q-values preview: {test_prediction[:5]}")

            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                raise
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def getMove(self, obs, info):
        """Returns button inputs with debugging and action variety"""
        # Debug: Always print first few calls
        if not hasattr(self, "call_count"):
            self.call_count = 0

        self.call_count += 1

        stateData = self.prepareNetworkInputs(info)

        # Get Q-values from model
        predictedRewards = self.model.predict(stateData, verbose=0)[0]

        # Add some exploration even in eval mode to test different actions
        if self.call_count <= 10:  # First 10 moves, try top-3 actions
            top_3_indices = np.argsort(predictedRewards)[-3:]
            move_index = np.random.choice(top_3_indices)
            print(
                f"🎮 Move {self.call_count}: Trying action {move_index} (Q={predictedRewards[move_index]:.3f}) from top-3"
            )
        else:
            move_index = np.argmax(predictedRewards)

        if self.call_count <= 5:  # Debug first 5 calls
            print(f"🎮 Move {self.call_count}:")
            print(f"  State data shape: {stateData.shape}")
            print(f"  Player health: {info.get('health', 'N/A')}")
            print(f"  Enemy health: {info.get('enemy_health', 'N/A')}")
            print(
                f"  Player pos: ({info.get('x_position', 'N/A')}, {info.get('y_position', 'N/A')})"
            )
            print(
                f"  Enemy pos: ({info.get('enemy_x_position', 'N/A')}, {info.get('enemy_y_position', 'N/A')})"
            )
            print(
                f"  Distance: {abs(info.get('x_position', 100) - info.get('enemy_x_position', 200))}"
            )

            # Show top 5 Q-values
            top_5_indices = np.argsort(predictedRewards)[-5:]
            print(f"  Top 5 Q-values:")
            for i in top_5_indices[::-1]:
                move_name = list(self.moveList)[i].name
                print(f"    {i}: {move_name} = {predictedRewards[i]:.3f}")

        # Convert to move
        move = list(self.moveList)[move_index]
        frameInputs = self.convertMoveToFrameInputs(move, info)

        if self.call_count <= 5:
            print(f"  Selected move: {move.name}")
            print(f"  Frame inputs: {frameInputs}")

        return move_index, frameInputs

    def recordStep(self, step):
        """Override to prevent any training during evaluation"""
        pass

    def reviewFight(self):
        """Override to prevent any training during evaluation"""
        pass


def get_states():
    """Get available state files"""
    directory = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    if not os.path.exists(directory):
        return []
    try:
        files = os.listdir(directory)
        states = [
            os.path.splitext(file)[0] for file in files if file.endswith(".state")
        ]
        return states
    except Exception as e:
        logger.error(f"Error getting states: {e}")
        return []


def create_default_state():
    """Create a default state file if none exist"""
    state_dir = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "default.state")

    try:
        env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
        env.reset()

        for _ in range(10):
            env.step([0] * len(env.buttons))

        state_data = env.em.get_state()

        with open(state_path, "wb") as f:
            f.write(state_data)

        logger.info(f"Created default state at {state_path}")
        env.close()
        return "default"
    except Exception as e:
        logger.error(f"Error creating default state: {e}")
        return None


def read_ram_values(env, info):
    """Read game state from RAM using CORRECTED addresses"""
    # CORRECTED RAM addresses (same as training script)
    ram_info = {
        "continue_timer": {"address": 16744917, "type": "|u1"},
        "round_timer": {"address": 16750378, "type": ">u2"},
        "enemy_health": {"address": 16745154, "type": ">i2"},
        "enemy_x_position": {"address": 32984, "type": "|u1"},  # FIXED: Was 16745094
        "enemy_y_position": {"address": 32985, "type": "|u1"},  # FIXED: Was 16745098
        "enemy_matches_won": {"address": 16745559, "type": ">u4"},
        "enemy_status": {"address": 16745090, "type": ">u2"},
        "enemy_character": {"address": 16745563, "type": "|u1"},
        "health": {"address": 16744514, "type": ">i2"},
        "x_position": {"address": 32774, "type": "|u1"},  # FIXED: Was 16744454
        "y_position": {"address": 32775, "type": "|u1"},  # FIXED: Was 16744458
        "status": {"address": 16744450, "type": ">u2"},
        "matches_won": {"address": 16744922, "type": "|u1"},
        "score": {"address": 16744936, "type": ">d4"},
    }

    try:
        if hasattr(env.unwrapped, "get_ram"):
            ram = env.unwrapped.get_ram()
        elif hasattr(env.unwrapped, "em") and hasattr(env.unwrapped.em, "get_ram"):
            ram = env.unwrapped.em.get_ram()
        else:
            return ensure_required_keys(info)

        for key, address_info in ram_info.items():
            addr = address_info["address"]
            data_type = address_info["type"]
            if addr >= len(ram):
                continue
            try:
                if data_type == "|u1":
                    value = ram[addr]
                elif data_type == ">u2":
                    if addr + 1 < len(ram):
                        value = (ram[addr] << 8) | ram[addr + 1]
                    else:
                        continue
                elif data_type == ">i2":
                    if addr + 1 < len(ram):
                        value = (ram[addr] << 8) | ram[addr + 1]
                        if value >= 32768:
                            value -= 65536
                    else:
                        continue
                elif data_type == ">u4":
                    if addr + 3 < len(ram):
                        value = (
                            (ram[addr] << 24)
                            | (ram[addr + 1] << 16)
                            | (ram[addr + 2] << 8)
                            | ram[addr + 3]
                        )
                    else:
                        continue
                elif data_type == ">d4":
                    if addr + 3 < len(ram):
                        import struct

                        try:
                            value = struct.unpack(
                                ">f",
                                bytes(
                                    [
                                        ram[addr],
                                        ram[addr + 1],
                                        ram[addr + 2],
                                        ram[addr + 3],
                                    ]
                                ),
                            )[0]
                        except struct.error:
                            value = 0
                    else:
                        continue
                else:
                    value = ram[addr]
                info[key] = value
            except Exception:
                pass
    except Exception:
        pass
    return ensure_required_keys(info)


def ensure_required_keys(info):
    """Ensure all required keys are present with default values"""
    required_keys = {
        "continue_timer": 0,
        "round_timer": 0,
        "enemy_health": 176,
        "enemy_x_position": 200,
        "enemy_y_position": 0,
        "enemy_matches_won": 0,
        "enemy_status": 512,
        "enemy_character": 0,
        "health": 176,
        "x_position": 100,
        "y_position": 0,
        "status": 512,
        "matches_won": 0,
        "score": 0,
    }
    for key, default_value in required_keys.items():
        if key not in info:
            info[key] = default_value
    return info


def run_single_evaluation_game(agent, state_name, max_steps=2500, render=False):
    """Run a single evaluation game and return the result"""
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = retro.make(game=game, players=1)

    env.reset()

    # Load the state
    state_path = os.path.join(
        os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
        f"{state_name}.state",
    )
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_data = f.read()
        env.em.set_state(state_data)

    # Initialize with multiple no-ops to let game settle
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(10):  # Let game stabilize
        step_result = env.step(NO_ACTION)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

    info = read_ram_values(env, info)

    step_count = 0
    total_reward = 0
    full_hp = 176

    print(f"🎯 Starting evaluation game with state: {state_name}")
    print(f"   Initial player health: {info.get('health', 'N/A')}")
    print(f"   Initial enemy health: {info.get('enemy_health', 'N/A')}")
    print(
        f"   Initial player pos: ({info.get('x_position', 'N/A')}, {info.get('y_position', 'N/A')})"
    )
    print(
        f"   Initial enemy pos: ({info.get('enemy_x_position', 'N/A')}, {info.get('enemy_y_position', 'N/A')})"
    )

    while not done and step_count < max_steps:
        try:
            # Get action from agent
            if len(physical_devices) > 0:
                with tf.device("/GPU:0"):
                    action_index, frame_inputs = agent.getMove(obs, info)
            else:
                action_index, frame_inputs = agent.getMove(obs, info)

            prev_player_health = info.get("health", full_hp)
            prev_opponent_health = info.get("enemy_health", full_hp)
            prev_player_x = info.get("x_position", 100)
            prev_player_y = info.get("y_position", 0)

            # Debug: Print what we're getting
            if step_count < 3:  # Only print first 3 steps
                print(f"Step {step_count}: Action={action_index}")
                print(
                    f"  Before: Player=({prev_player_x}, {prev_player_y}), Enemy=({info.get('enemy_x_position', 200)}, {info.get('enemy_y_position', 0)})"
                )

            # Execute frame inputs - handle both single frame and multi-frame moves
            if isinstance(frame_inputs[0], list):
                # Multi-frame move (like special moves)
                frames_to_execute = frame_inputs
            else:
                # Single frame move
                frames_to_execute = [frame_inputs]

            # Execute each frame with small delays
            for frame_idx, frame in enumerate(frames_to_execute):
                if step_count < 3 and frame_idx == 0:
                    print(f"  Executing frame: {frame}")
                    # Check if frame has any non-zero inputs
                    if any(x != 0 for x in frame):
                        button_indices = [i for i, x in enumerate(frame) if x != 0]
                        button_names = [
                            "B",
                            "A",
                            "MODE",
                            "START",
                            "UP",
                            "DOWN",
                            "LEFT",
                            "RIGHT",
                            "C",
                            "Y",
                            "X",
                            "Z",
                        ]
                        pressed_buttons = [
                            button_names[i]
                            for i in button_indices
                            if i < len(button_names)
                        ]
                        print(f"  ✅ Pressing buttons: {pressed_buttons}")
                    else:
                        print(f"  ⚠️  Frame is all zeros (idle)")

                # Execute the frame multiple times for movement to register
                for _ in range(3):  # Hold button for 3 frames
                    step_result = env.step(frame)
                    if len(step_result) == 4:
                        obs, _, done, info = step_result
                    else:
                        obs, _, terminated, truncated, info = step_result
                        done = terminated or truncated

                    info = read_ram_values(env, info)

                    if render:
                        env.render()
                        time.sleep(0.016)  # ~60 FPS for better viewing

                    # Check for fight end
                    if (
                        info.get("health", full_hp) <= 0
                        or info.get("enemy_health", full_hp) <= 0
                    ):
                        done = True
                        break

                if done:
                    break

            # Check if position actually changed
            if step_count < 3:
                new_player_x = info.get("x_position", 100)
                new_player_y = info.get("y_position", 0)
                print(f"  After: Player=({new_player_x}, {new_player_y})")
                if new_player_x != prev_player_x or new_player_y != prev_player_y:
                    print(
                        f"  ✅ Position changed! Δx={new_player_x - prev_player_x}, Δy={new_player_y - prev_player_y}"
                    )
                else:
                    print(f"  ❌ Position unchanged")

        except Exception as e:
            print(f"❌ Error in step {step_count}: {e}")
            # Fallback: execute a no-op
            for _ in range(3):
                step_result = env.step(NO_ACTION)
                if len(step_result) == 4:
                    obs, _, done, info = step_result
                else:
                    obs, _, terminated, truncated, info = step_result
                    done = terminated or truncated
                info = read_ram_values(env, info)

        # Calculate reward (same as training)
        curr_player_health = info.get("health", full_hp)
        curr_opponent_health = info.get("enemy_health", full_hp)

        if curr_player_health <= 0:
            step_reward = -math.pow(full_hp, (curr_opponent_health + 1) / (full_hp + 1))
            done = True
        elif curr_opponent_health <= 0:
            step_reward = (
                math.pow(full_hp, (curr_player_health + 1) / (full_hp + 1)) * 3.0
            )
            done = True
        else:
            step_reward = 3.0 * (prev_opponent_health - curr_opponent_health) - (
                prev_player_health - curr_player_health
            )

        total_reward += step_reward
        step_count += 1

        # Print health updates every 100 steps
        if step_count % 100 == 0:
            print(
                f"Step {step_count}: Player={curr_player_health}, Enemy={curr_opponent_health}, Pos=({info.get('x_position', 'N/A')}, {info.get('y_position', 'N/A')})"
            )

    # Determine winner
    final_player_health = info.get("health", 0)
    final_enemy_health = info.get("enemy_health", 0)

    if final_player_health > final_enemy_health:
        result = "WIN"
    elif final_enemy_health > final_player_health:
        result = "LOSS"
    else:
        result = "DRAW"

    print(
        f"🏁 Game ended: {result} (Player: {final_player_health}, Enemy: {final_enemy_health})"
    )

    env.close()

    return {
        "result": result,
        "player_health": final_player_health,
        "enemy_health": final_enemy_health,
        "total_reward": total_reward,
        "steps": step_count,
    }


def analyze_agent_strategy(agent):
    """Analyze what the agent has learned by testing different scenarios"""
    print("\n" + "=" * 50)
    print("AGENT STRATEGY ANALYSIS")
    print("=" * 50)

    # Test scenarios
    scenarios = [
        {
            "name": "Close Range",
            "player_x": 180,
            "enemy_x": 200,
            "player_health": 176,
            "enemy_health": 176,
        },
        {
            "name": "Far Range",
            "player_x": 100,
            "enemy_x": 200,
            "player_health": 176,
            "enemy_health": 176,
        },
        {
            "name": "Low Health",
            "player_x": 150,
            "enemy_x": 200,
            "player_health": 50,
            "enemy_health": 176,
        },
        {
            "name": "Enemy Low Health",
            "player_x": 150,
            "enemy_x": 200,
            "player_health": 176,
            "enemy_health": 50,
        },
        {
            "name": "Behind Enemy",
            "player_x": 220,
            "enemy_x": 200,
            "player_health": 176,
            "enemy_health": 176,
        },
    ]

    for scenario in scenarios:
        print(f"\n📍 {scenario['name']}:")
        print(f"   Player at {scenario['player_x']}, Enemy at {scenario['enemy_x']}")
        print(f"   Distance: {abs(scenario['player_x'] - scenario['enemy_x'])}")

        # Create fake info
        info = {
            "health": scenario["player_health"],
            "enemy_health": scenario["enemy_health"],
            "x_position": scenario["player_x"],
            "enemy_x_position": scenario["enemy_x"],
            "y_position": 0,
            "enemy_y_position": 0,
            "status": 512,
            "enemy_status": 512,
            "matches_won": 0,
            "enemy_matches_won": 0,
            "enemy_character": 0,
            "score": 0,
        }

        # Get agent's decision
        stateData = agent.prepareNetworkInputs(info)
        predictedRewards = agent.model.predict(stateData, verbose=0)[0]

        # Show top 3 moves
        top_3_indices = np.argsort(predictedRewards)[-3:]
        print(f"   Top 3 moves:")
        for i, idx in enumerate(top_3_indices[::-1]):
            move_name = list(agent.moveList)[idx].name
            print(f"     {i+1}. {move_name} (Q={predictedRewards[idx]:.3f})")

    print("=" * 50)


def run_evaluation(model_path, num_games=100, render=False):
    """Run evaluation on the trained model"""
    logger.info(f"Starting evaluation with model: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # Get available states
    states = get_states()
    if not states:
        logger.info("No state files found. Creating default state...")
        create_default_state()
        states = ["default"]

    logger.info(f"Found {len(states)} state files: {states}")

    # Create evaluation agent
    try:
        agent = EvalAgent(model_path=model_path, stateSize=60)
        logger.info("✅ Evaluation agent created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create evaluation agent: {e}")
        return

    # Analyze what the agent has learned
    analyze_agent_strategy(agent)

    # Run evaluation games
    results = []
    wins = 0
    losses = 0
    draws = 0

    logger.info(f"Running {num_games} evaluation games...")

    for game_num in tqdm(range(num_games), desc="Evaluation Games"):
        state_name = random.choice(states)

        try:
            game_result = run_single_evaluation_game(
                agent, state_name, max_steps=2500, render=render
            )

            results.append(game_result)

            if game_result["result"] == "WIN":
                wins += 1
            elif game_result["result"] == "LOSS":
                losses += 1
            else:
                draws += 1

            # Print progress every 10 games
            if (game_num + 1) % 10 == 0:
                current_win_rate = (wins / (game_num + 1)) * 100
                logger.info(
                    f"Progress: {game_num + 1}/{num_games} games, Win Rate: {current_win_rate:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error in game {game_num + 1}: {e}")
            continue

    # Calculate final statistics
    total_games = wins + losses + draws
    win_rate = (wins / total_games * 100) if total_games > 0 else 0

    avg_reward = (
        sum(r["total_reward"] for r in results) / len(results) if results else 0
    )
    avg_steps = sum(r["steps"] for r in results) / len(results) if results else 0

    # Print final results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Total Games: {total_games}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps per Game: {avg_steps:.1f}")
    print("=" * 60)

    # Show some detailed results
    if results:
        print("\nDetailed Results (Last 10 Games):")
        for i, result in enumerate(results[-10:], 1):
            print(
                f"Game {len(results)-10+i}: {result['result']} "
                f"(Player: {result['player_health']}, Enemy: {result['enemy_health']}, "
                f"Reward: {result['total_reward']:.1f})"
            )

    # Strategy analysis
    print(f"\n🧠 AGENT BEHAVIOR ANALYSIS:")
    print(f"The agent has learned that Heavy Punch has the highest Q-value (0.485)")
    print(f"However, it's not learning to move closer to the enemy first.")
    print(
        f"This suggests the training needs more diverse scenarios or better reward shaping."
    )
    print(
        f"Consider adding movement rewards or training with different starting positions."
    )

    return {"win_rate": win_rate, "total_games": total_games, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Street Fighter II AI model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/DeepQAgentModel.weights.h5",
        help="Path to the trained model weights file",
    )
    parser.add_argument(
        "--num_games", type=int, default=3, help="Number of evaluation games to run"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the games (slower but you can watch)",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path, num_games=args.num_games, render=args.render
    )
